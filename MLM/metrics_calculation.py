
import os
import sys
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
# sys.path.append('/kaggle/working/SRLPredictionEasel')
sys.path.append('../')
from infer_pipeline import inferPipeline
from data_maker import DataMaker
from data_preparation import * 
from mlm_utils.transform_func import get_files, check_data_dir, get_idx_arg_preds
from mlm_utils.metric_func import influence_score, relevance_score, brier_score_multi_class, competence_score
from scipy.stats import spearmanr
from datetime import datetime
from logger_ import make_logger

def evaluateWord(dataDir, file_name, model, labelRn, wrtDir=None, hasTrueLabels=True, needMetrics=True, is_mask_token=False, del_mask_token=False):
    dataMaker = DataMaker(
        data_file= os.path.join(dataDir, file_name)
    )
    
    result = dataMaker.evaluate(model, labelRn, wrtPredPath=file_name, wrtDir=wrtDir, returnPreds=True, hasTrueLabels=hasTrueLabels, needMetrics=needMetrics, is_mask_token=is_mask_token, del_mask_token=del_mask_token)
    return {
        'uid': result[0],
        'pred': result[1],
        'logitsSoftmax': result[2],
        'logitsRaw': result[3],
        'label': result[4] # masked word kh cos label
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
        

def get_importance_score(origin_data, perturbed_data, labelMap):
    '''
    Calculate influence, relevance score and brier_score for each perturbation affected argument.
    Save the result in a dictionary with key is the changed argument, value is a list of important scores.
    Args:
        origin_data: dict, contains the original data
        perturbed_data: dict, contains the masked data
        labelMap: dict, label mapping
        
    Returns:
        score_dict: dict, key is the changed argument, value is a list of important scores
        list_pair_score: list, contains the score of each pair of origin and perturbed data
    '''
    score_dict = {}
    list_pair_score = []
    for i in range(len(origin_data['uid'])):
        for j in range(len(perturbed_data['uid'])):
            if int(origin_data['uid'][i]) == int(perturbed_data['uid'][j]):
                changed_args = check_arg_change(perturbed_data['pred'][j], perturbed_data['label'][j])
                inf_score, w_inf = influence_score(origin_data['logitsSoftmax'][i], perturbed_data['logitsSoftmax'][j], get_idx_arg_preds(origin_data['pred'][i], perturbed_data['pred'][j]))
                
                rel_score, w_rel = relevance_score(origin_data['logitsSoftmax'][i], perturbed_data['logitsSoftmax'][j],labelMap, origin_data['label'][i], origin_data['pred'][i], perturbed_data['pred'][j])
                
                brier_score = (1 - brier_score_multi_class(origin_data['label'][i], origin_data['logitsSoftmax'][i], labelMap))
                score = {
                        'uid': origin_data['uid'][i],
                        'influence': round(sum(inf_score) / sum(w_inf), 5) if sum(w_inf) != 0 else 0,
                        'relevance': round(sum(rel_score) / sum(w_rel), 5) if sum(w_rel) != 0 else 0,
                        'brier_score': round(brier_score, 5)
                        }
                list_pair_score.append(score)
                
                # save to score_dict with key is item in changed_args, value is score
                for arg in changed_args:
                    if arg not in score_dict:
                        score_dict[arg] = []
                    score_dict[arg].append(score)
    return score_dict, list_pair_score  


def get_comp_each_arg(dataMaskedDir, dataOriginDir, model, labelRn, logger, wriDir, is_mask_token, del_mask_token):
    file_mask = sorted(get_files(dataMaskedDir))
    file_origin = sorted(get_files(dataOriginDir))
   
    list_comp_dict = []
    for mask, origin in zip(file_mask, file_origin):
        logger.info("Calculate file {} and {}".format(mask, origin))
        resultwordMasked = evaluateWord(dataMaskedDir, mask, model, labelRn, hasTrueLabels=False, needMetrics=False, is_mask_token=is_mask_token, del_mask_token=del_mask_token)
        resultOrigin = evaluateWord(dataOriginDir, origin, model, labelRn, hasTrueLabels=True, needMetrics=False)

        labelMap = {v: k for k, v in labelRn.items()}
        comp_score, list_pair_score = get_importance_score(resultOrigin, resultwordMasked, labelMap)
       
        # save list_pair_score to csv if wriDir is not None
        if wriDir is not None:
            check_data_dir(wriDir, auto_create=True)
            df_score = pd.DataFrame(list_pair_score)
            df_score.to_csv(os.path.join(wriDir, 'pair_score_{}.csv'.format(mask.replace('.json', ''))), sep = "\t", index = False)
        
        comp_dict = {}
        for key, value in comp_score.items():
            logger.info("key: {} has {} sentences, competence {}, brier_score_loss {}, with p-value {}." \
                        .format(key, len(value), competence_score(value)[0], competence_score(value)[2], competence_score(value)[1]))
           
            comp_dict[key] = {'comp': competence_score(value)[0],
                              'brier_score': competence_score(value)[2]}
        logger.info("-------------------------------------------------")
        list_comp_dict.append(comp_dict)
       
    return list_comp_dict

# using plot to visualize the result
def plot_corr(comp_list, brier_score_list, save_img=False, save_path=None):
    
    plt.scatter(comp_list, brier_score_list)
   
    plt.xlabel('Competence')
    plt.ylabel('1 - Brier Score')
    plt.title('Correlation between Competence vs (1 - Brier Score)')
  
    # Hiển thị biểu đồ
    plt.grid(True)
    
    plt.show()
    if save_img and save_path is not None:
        plt.savefig(save_path)

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
    parser.add_argument('--wriDir', type=Path,
                        help="path to the directory where the scores will be written.")
    parser.add_argument('--is_mask_token', type=bool, default=False)
    parser.add_argument('--del_mask_token', type=bool, default=False)
    args = parser.parse_args()
    
    # setting logging
    now = datetime.now()
    logDir = now.strftime("%d_%m-%H_%M")
    if not os.path.isdir(logDir):
        os.makedirs(logDir)

    logger = make_logger(name = args.log_name, debugMode=True,
                        logFile=os.path.join(logDir, '{}.log'.format(args.log_name)), silent=True)
    logger.info("logger created.")
    

    pipe = inferPipeline(logger, args.model_path)
    labelMap = pipe.taskParams.labelMap['conllsrl']
    labelRn = {v:k for k,v in labelMap.items()}
    
    list_comp_dict = get_comp_each_arg(args.data_mask_dir, args.data_origin_dir, pipe.model, labelRn, logger, args.wriDir, is_mask_token=args.is_mask_token, del_mask_token=args.del_mask_token)
   
   
    # Task 1.2: correlation between competence and brier score
    comp_list = []
    brier_score_list = []
    for entry in list_comp_dict:
        for key, value in entry.items():
            comp_list.append(value['comp'])
            brier_score_list.append(value['brier_score'])

    corr, p_value  = spearmanr(comp_list, brier_score_list)
    logger.info("Spearman Correlation Coefficient: {}, with p-value {}.".format(corr, p_value))

    plot_corr(comp_list, brier_score_list, save_img=True, save_path=os.path.join(logDir, 'img_{}.png'.format(args.log_name)))
    logger.info("Done Visualization.")
    

if __name__ == '__main__':
    main()
        