
import os
import sys
from matplotlib import pyplot as plt
import torch
import numpy as np 
from pathlib import Path
sys.path.append('/kaggle/working/SRLPredictionEasel')
from SRL.model import multiTaskModel
from data_maker import DataMaker
from data_preparation import * 
from mlm_utils.transform_func import get_files
from mlm_utils.metric_func import influence_score, relevance_score, brier_score_multi_class, competence_score, get_idx_arg_preds
from scipy.stats import spearmanr
from datetime import datetime
from logger_ import make_logger


def load_params(model_file):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load finetuned model 
    loadedDict = torch.load(model_file, map_location=torch.device(device))
    taskParams = loadedDict['task_params']
    allParams = {}
    allParams['task_params'] = taskParams
    allParams['gpu'] = torch.cuda.is_available()
    allParams['num_train_steps'] = 10
    allParams['warmup_steps'] = 0
    allParams['learning_rate'] = 2e-05
    allParams['epsilon'] = 1e-8

    # modelName = taskParams.modelType.name.lower()
    # print("Model Name: ", modelName)
    # _, _ , tokenizerClass, defaultName = NLP_MODELS[modelName]
    # configName = taskParams.modelConfig
    # if configName is None:
    #     configName = defaultName

    return allParams, loadedDict

def get_word(dataDir, file_name, model, labelRn, wrtDir=None, hasTrueLabels=True, needMetrics=True):
    dataMaker = DataMaker(
        data_file= os.path.join(dataDir, file_name)
    )
    
    result = dataMaker.evaluate(model, labelRn, wrtPredPath=file_name, wrtDir=wrtDir, returnPreds=True, hasTrueLabels=hasTrueLabels, needMetrics=needMetrics)
    return {
        'uid': result[0],
        'pred': result[1],
        'score': result[2],
        'logitsSoftmax': result[3],
        'logitsRaw': result[4],
        'label': result[5] # masked word kh cos label
    }
    
def check_arg_change(preds_masked, label):
    assert len(preds_masked) == len(label), 'Length of preds_masked and label must be the same'
    changed_args = set()
    
    # check if preds_masked is different from label, then save the value of label. Remember save the unique value
    for i in range(len(preds_masked)):
        if preds_masked[i] != label[i] and (label[i].startswith('B-A') or label[i].startswith('I-A')):
            if (label[i].startswith('B-A') or label[i].startswith('I-A')):
                changed_args.add(label[i].split('-')[-1])    
    
    return changed_args    
        

def get_pair_inf_rel(origin, masked, labelMap):
    comp_dict = {}
    for i in range(len(origin['uid'])):
        for j in range(len(masked['uid'])):
            if int(origin['uid'][i]) == int(masked['uid'][j]):
                
                changed_args = check_arg_change(masked['pred'][j], masked['label'][j])
                inf_score, w_inf = influence_score(origin['logitsSoftmax'][i], masked['logitsSoftmax'][j], get_idx_arg_preds(origin['pred'][i], masked['pred'][j]))
                rel_score, w_rel = relevance_score(origin['logitsSoftmax'][i], masked['logitsSoftmax'][j],labelMap, origin['label'][i], origin['pred'][i], masked['pred'][j])
                
                brier_score = (1 - brier_score_multi_class(origin['label'][i], origin['logitsSoftmax'][i], labelMap))
                score = {
                        'uid': origin['uid'][i],
                        'influence': sum(inf_score) / sum(w_inf) if sum(w_inf) != 0 else 0,
                        'relevance': sum(rel_score) / sum(w_rel) if sum(w_rel) != 0 else 0,
                        'brier_score': brier_score
                        }
                
                # save to comp_dict with key is item in changed_args, value is score
                for arg in changed_args:
                    if arg not in comp_dict:
                        comp_dict[arg] = []
                    comp_dict[arg].append(score)
           
    return comp_dict


def get_comp_each_arg(dataMaskedDir, dataOriginDir, model, labelRn, logger):
    file_mask = sorted(get_files(dataMaskedDir))
    file_origin = sorted(get_files(dataOriginDir))
    
    list_spearrman_dict = []
    for mask, origin in zip(file_mask, file_origin):
        logger.info("Calculate file {} and {}".format(mask, origin))
        resultwordMasked = get_word(dataMaskedDir, mask, model, labelRn, hasTrueLabels=False, needMetrics=False)
        resultOrigin = get_word(dataOriginDir, origin, model, labelRn, hasTrueLabels=True, needMetrics=False)

        labelMap = {v: k for k, v in labelRn.items()}
        comp_score = get_pair_inf_rel(resultOrigin, resultwordMasked, labelMap)
       
        spearmanr_dict = {}
        for key, value in comp_score.items():
            logger.info("key: {} has {} sentences, competence {}, brier_score_loss {}, with p-value {}." \
                        .format(key, len(value), competence_score(value)[0], competence_score(value)[2], competence_score(value)[1]))
           
            spearmanr_dict[key] = {'comp': competence_score(value)[0],
                                   'brier_score': competence_score(value)[2]}
        logger.info("-------------------------------------------------")
        list_spearrman_dict.append(spearmanr_dict)
    return list_spearrman_dict

# using plot to visualize the result
def plot_corr(comp_list, brier_score_list, save_img=False, save_path=None):
    
    plt.scatter(comp_list, brier_score_list)
    plt.xlabel('Competence')
    plt.ylabel('Brier Score')
    plt.title('Correlation between Competence vs Brier Score')
    plt.show()
    if save_img and save_path is not None:
        plt.savefig('{}.png'.format(save_path))


def main():
    # taking in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_mask_dir', type=Path, required=True,  
                        default=Path("data_mlm/perturbed_data/avg_neg_cos/"), 
                        help="path to the masked data directory.")
    parser.add_argument('--data_origin_dir', type=Path, required=True, 
                        default=Path('data_mlm/process_folder/coNLL_tsv_json/ner_json/'),
                        help="path to the original data directory.")
    parser.add_argument('--model_path', type=Path, default=Path('output/multi_task_model_9_13050.pt'),
                        help="path to the model file.")
    parser.add_argument('--log_name', type=str, default = 'cal_comp_by_arg',
                        help = "name of the log file to be created.")
   
    args = parser.parse_args()
    
    
    # setting logging
    now = datetime.now()
    logDir = now.strftime("%d_%m-%H_%M")
    if not os.path.isdir(logDir):
        os.makedirs(logDir)

    logger = make_logger(name = args.log_name, debugMode=True,
                        logFile=os.path.join(logDir, '{}.log'.format(args.log_name)), silent=True)
    logger.info("logger created.")

    
    a = load_params(args.model_path)
    labelMap = a[0]['task_params'].labelMap['conllsrl']
    labelRn = {v:k for k,v in labelMap.items()}
    
    model = multiTaskModel(a[0])
    model.load_multi_task_model(a[1])
    logger.info('saved model loaded with global step {} from {}'.format(model.globalStep,
                                                                            args.model_path))
    list_spearmanr_dict = get_comp_each_arg(args.data_mask_dir, args.data_origin_dir, model, labelRn, logger)
    
    comp_list = []
    brier_score_list = []
    for entry in list_spearmanr_dict:
        for key, value in entry.items():
            comp_list.append(value['comp'])
            brier_score_list.append(value['brier_score'])


    corr, p_value  = spearmanr(comp_list, brier_score_list)
    logger.info("Spearman Correlation Coefficient: {}, with p-value {}".format(corr, p_value))

    plot_corr(comp_list, brier_score_list, save_img=True, save_path=args.log_name)
    logger.info("Done Visualization.")
if __name__ == '__main__':
    main()
        