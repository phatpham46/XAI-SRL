import ast
from sklearn.metrics import brier_score_loss
from sklearn.preprocessing import label_binarize
import numpy as np
from scipy.stats import spearmanr
from mlm_utils.transform_func import get_idx_arg_preds

def cosine_sim(a, b):
    '''
    Function to calculate cosine similarity between two vectors.
    '''
    if not isinstance(a, list):
        a = ast.literal_eval(a)
    if not isinstance(b, list):
        b = ast.literal_eval(b)
            
    return round(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), 5)


def cosine_module(a, b, cosine_sum):
    '''
    Function to calculate cosine similarity between two vectors
    '''
    norm_array1 = np.linalg.norm(a)
    norm_array2 = np.linalg.norm(b)
    
    module_similarity = 1 - (np.abs(norm_array1 - norm_array2) / (norm_array1 + norm_array2))
    
    return round(module_similarity * cosine_sum, 5)


def influence_score(outLogitsSigmoid_original, outLogitsSigmoid_meddle, list_arg_change):
    influence_score = []
    weight = []
    if len(list_arg_change) == 0:
        return influence_score, weight
    else:
        for i in range(min(len(outLogitsSigmoid_original), len(outLogitsSigmoid_meddle))):
            if i not in list_arg_change:
                continue
            max_index_original = np.argmax(outLogitsSigmoid_original[i])
        
            max_index_meddle = np.argmax(outLogitsSigmoid_meddle[i])
           
            if max_index_original == max_index_meddle:
              
                influence_score.append((outLogitsSigmoid_original[i][max_index_original] - outLogitsSigmoid_meddle[i][max_index_meddle]) / max(outLogitsSigmoid_original[i][max_index_original], outLogitsSigmoid_meddle[i][max_index_meddle]))
                weight.append(1)
            
            else:
                influ_old_label = (outLogitsSigmoid_original[i][max_index_original] - outLogitsSigmoid_meddle[i][max_index_original]) / max(outLogitsSigmoid_original[i][max_index_original], outLogitsSigmoid_meddle[i][max_index_original])
                influ_new_label = (outLogitsSigmoid_meddle[i][max_index_meddle] - outLogitsSigmoid_original[i][max_index_meddle]) / max(outLogitsSigmoid_original[i][max_index_meddle], outLogitsSigmoid_meddle[i][max_index_meddle])
                influence_score.append(influ_old_label + influ_new_label)
                weight.append(2)
    return influence_score, weight


def relevance_score(prob_origin_, prob_masked_, labMap, label_gold, label_origin, label_masked):
    relevance = []
    weight = []
    jud_space = get_idx_arg_preds(label_origin, label_masked, label_gold)
    
    for i in range(len(prob_origin_)):
        if i not in jud_space:
            continue
       
        max_index_origin = np.argmax(prob_origin_[i])
        max_index_masked = np.argmax(prob_masked_[i])
       
        idx_label_gold = labMap[str(label_gold[i])]
        
        if label_gold[i] == label_origin[i] and label_gold[i] == label_masked[i]:
            score_increase_gold = (prob_origin_[i][idx_label_gold] - prob_masked_[i][idx_label_gold])/max(prob_origin_[i][idx_label_gold], prob_masked_[i][idx_label_gold])
            relevance.append(score_increase_gold)
            weight.append(1)
           
        elif label_masked[i] != label_origin[i] and label_origin[i] == label_gold[i]:
            score_increase_gold = (prob_origin_[i][idx_label_gold] - prob_masked_[i][idx_label_gold])/max(prob_origin_[i][idx_label_gold], prob_masked_[i][idx_label_gold])
            score_decrease_mask = (prob_masked_[i][max_index_masked] - prob_origin_[i][max_index_masked])/max(prob_masked_[i][max_index_masked], prob_origin_[i][max_index_masked])
            relevance.append((score_increase_gold + score_decrease_mask)/2)
            weight.append(2)
           
        elif label_origin[i] != label_masked[i] and label_masked[i] == label_gold[i]:
            score_increase_gold = (prob_origin_[i][idx_label_gold] - prob_masked_[i][idx_label_gold])/max(prob_origin_[i][idx_label_gold], prob_masked_[i][idx_label_gold])
            score_decrease_origin = (prob_masked_[i][max_index_origin] - prob_origin_[i][max_index_origin])/max(prob_masked_[i][max_index_origin], prob_origin_[i][max_index_origin])
            relevance.append((score_increase_gold + score_decrease_origin)/2)
            weight.append(2)
           
        elif label_gold[i] != label_origin[i] and label_masked[i] != label_gold[i]:
            score_increase_gold = (prob_origin_[i][idx_label_gold] - prob_masked_[i][idx_label_gold])/max(prob_origin_[i][idx_label_gold], prob_masked_[i][idx_label_gold])
            score_decrease_mask = (prob_masked_[i][max_index_masked] - prob_origin_[i][max_index_masked])/max(prob_masked_[i][max_index_masked], prob_origin_[i][max_index_masked])
            score_decrease_origin = (prob_masked_[i][max_index_origin] - prob_origin_[i][max_index_origin])/max(prob_masked_[i][max_index_origin], prob_origin_[i][max_index_origin])
            relevance.append((score_increase_gold + score_decrease_mask + score_decrease_origin)/3)
            weight.append(1)
           
    return relevance, weight


def brier_score_multi_class(y_true, y_prob, labelMap):
    """
    Calculate the Brier score for multi-class classification using scikit-learn.

    Parameters:
    y_true (numpy.ndarray): True class labels, shape (n_samples,)
    y_prob (numpy.ndarray): Predicted probabilities, shape (n_samples, n_classes)

    Returns:
    float: Brier score
    """
    
    y_prob = np.array(y_prob, dtype=float)  # Predicted probabilities
    
    # Ensure y_true is a 1D array
    y_true = np.array(y_true)
    
    # label map 
    y_true = [labelMap[item] for item in y_true]
    y_true_one_hot = label_binarize(y_true, classes=np.arange(y_prob.shape[1]))
  
    # Calculate the Brier score for each class and average them
    brier_scores = np.array([brier_score_loss(y_true_one_hot[:, i], y_prob[:, i], pos_label=1) for i in range(y_prob.shape[1])])
    return np.mean(brier_scores) # mean for all classes


def competence_score(comp):
    influence_values = [abs(item['influence']) for item in comp]
    relevance_values = [item['relevance'] for item in comp]

    # Calculate Spearman correlation
    correlation_coefficient, p_value = spearmanr(influence_values, relevance_values)
    # get brier score for each unique uid in comp
    brier_score = np.mean([item['brier_score'] for item in comp])
    return round(correlation_coefficient, 4), round(p_value, 4), round(brier_score, 4)
