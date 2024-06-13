
import os
import sys
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
sys.path.append('/kaggle/working/SRLPredictionEasel')
# sys.path.append('../')
from infer_pipeline import inferPipeline
from data_maker import DataMaker
from data_preparation import * 
from mlm_utils.transform_func import get_files
from mlm_utils.metric_func import influence_score, relevance_score, brier_score_multi_class, competence_score, get_idx_arg_preds, corr_inf_lhs
from scipy.stats import spearmanr
from datetime import datetime
from logger_ import make_logger


def get_word(dataDir, file_name, model, labelRn, wrtDir=None, hasTrueLabels=True, needMetrics=True, is_mask_token=False, del_mask_token=False):
    dataMaker = DataMaker(
        data_file= os.path.join(dataDir, file_name)
    )
    
    result = dataMaker.evaluate(model, labelRn, wrtPredPath=file_name, wrtDir=wrtDir, returnPreds=True, hasTrueLabels=hasTrueLabels, needMetrics=needMetrics, is_mask_token=is_mask_token, del_mask_token=del_mask_token)
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
    '''
    Calculate influence and relevance score for each sentence in origin and masked data.
    Save the result in a dictionary with key is the changed argument, value is a list of scores.'''
    comp_dict = {}
    list_comp_score = []
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
                list_comp_score.append(score)
                # save to comp_dict with key is item in changed_args, value is score
                for arg in changed_args:
                    if arg not in comp_dict:
                        comp_dict[arg] = []
                    comp_dict[arg].append(score)
           
    return comp_dict, list_comp_score


def get_comp_each_arg(dataMaskedDir, dataOriginDir, model, labelRn, logger, is_mask_token, del_mask_token):
    file_mask = sorted(get_files(dataMaskedDir))
    file_origin = sorted(get_files(dataOriginDir))
    all_comp_score = []
    list_spearrman_dict = []
    for mask, origin in zip(file_mask, file_origin):
        logger.info("Calculate file {} and {}".format(mask, origin))
        resultwordMasked = get_word(dataMaskedDir, mask, model, labelRn, hasTrueLabels=False, needMetrics=False, is_mask_token=is_mask_token, del_mask_token=del_mask_token)
        resultOrigin = get_word(dataOriginDir, origin, model, labelRn, hasTrueLabels=True, needMetrics=False)

        labelMap = {v: k for k, v in labelRn.items()}
        comp_score, list_comp_score = get_pair_inf_rel(resultOrigin, resultwordMasked, labelMap)
        
        
        all_comp_score.extend(list_comp_score)
        spearmanr_dict = {}
        for key, value in comp_score.items():
            logger.info("key: {} has {} sentences, competence {}, brier_score_loss {}, with p-value {}." \
                        .format(key, len(value), competence_score(value)[0], competence_score(value)[2], competence_score(value)[1]))
           
            spearmanr_dict[key] = {'comp': competence_score(value)[0],
                                   'brier_score': competence_score(value)[2]}
        logger.info("-------------------------------------------------")
        list_spearrman_dict.append(spearmanr_dict)
        
    return list_spearrman_dict, all_comp_score

# using plot to visualize the result
def plot_corr(comp_list, brier_score_list, save_img=False, save_path=None):
    
    plt.scatter(comp_list, brier_score_list)
    plt.xlim(-1, 1)
    plt.ylim(0, 1)
    plt.xlabel('Competence')
    plt.ylabel('Brier Score')
    plt.title('Correlation between Competence vs Brier Score')
    
    # Thêm trục phân cách (trục x và y đi qua điểm (0, 0))
    plt.axhline(y=0, color='k', linestyle=':')  # Đường ngang tại y=0
    plt.axvline(x=0, color='k', linestyle=':')  # Đường dọc tại x=0

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
    parser.add_argument('--is_mask_token', type=bool, default=False )
    parser.add_argument('--del_mask_token', type=bool, default=False )
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
    
    list_spearmanr_dict, all_inf_score = get_comp_each_arg(args.data_mask_dir, args.data_origin_dir, pipe.model, labelRn, logger, is_mask_token=args.is_mask_token, del_mask_token=args.del_mask_token)
    
    # # Task 1.2: correlation between competence and brier score
    # comp_list = []
    # brier_score_list = []
    # for entry in list_spearmanr_dict:
    #     for key, value in entry.items():
    #         comp_list.append(value['comp'])
    #         brier_score_list.append(value['brier_score'])

    # corr, p_value  = spearmanr(comp_list, brier_score_list)
    # logger.info("Spearman Correlation Coefficient: {}, with p-value {}.".format(corr, p_value))

    # plot_corr(comp_list, brier_score_list, save_img=True, save_path=os.path.join(logDir, 'img_{}.png'.format(args.log_name)))
    # logger.info("Done Visualization.")
    

if __name__ == '__main__':
    main()
        