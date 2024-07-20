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
    
def cosine_module(a, b, cosine_sim):
    '''
    Function to calculate cosine similarity between two vectors
    '''
    norm_array1 = np.linalg.norm(a)
    norm_array2 = np.linalg.norm(b)
    
    module_similarity = 1 - (np.abs(norm_array1 - norm_array2) / (norm_array1 + norm_array2))
    
    return round(module_similarity * cosine_sim, 5)

def ele_wise_sub(a, b, negate=False):
    '''
    Function to calculate element-wise subtraction between two vectors.
    '''
    if not isinstance(a, list):
        a = ast.literal_eval(a)
    if not isinstance(b, list):
        b = ast.literal_eval(b)
    if len(a) != len(b):
        raise ValueError("Both lists must be of the same length")
   
    # Calculate the absolute differences
    abs_diff = [abs(i - j) for i, j in zip(a, b)]
    
    # Calculate the average of the absolute differences
    avg_abs_diff = sum(abs_diff) / len(abs_diff)
    
    # Return the result, negated if required
    if negate:
        return round(-avg_abs_diff, 5)
    else:
        return round(avg_abs_diff, 5)
    
def influence_score(logit_origin, logit_perturb, list_arg_change):
    '''
    Calculate the influence score for one perturbation.
    
    Args
        logit_origin: list of logits of model for the original input
        logit_perturb: list of logits of model for the perturb input
        list_arg_change: list of indices of the argument change
        
    Returns
        inf_score: list of influence score for each argument change
    
    '''
    inf_score = []
    inf_w = []
    if len(list_arg_change) == 0:
        return inf_score, inf_w
    else:
        for i in range(min(len(logit_origin), len(logit_perturb))):
            if i not in list_arg_change:
                continue
            max_idx_origin = np.argmax(logit_origin[i])
        
            max_idx_perturb = np.argmax(logit_perturb[i])
           
            if max_idx_origin == max_idx_perturb:
              
                inf_score.append((logit_origin[i][max_idx_origin] - logit_perturb[i][max_idx_perturb]) / max(logit_origin[i][max_idx_origin], logit_perturb[i][max_idx_perturb]))
                inf_w.append(1)
            
            else:
                inf_old_label = (logit_origin[i][max_idx_origin] - logit_perturb[i][max_idx_origin]) / max(logit_origin[i][max_idx_origin], logit_perturb[i][max_idx_origin])
                inf_new_label = (logit_perturb[i][max_idx_perturb] - logit_origin[i][max_idx_perturb]) / max(logit_origin[i][max_idx_perturb], logit_perturb[i][max_idx_perturb])
                inf_score.append(inf_old_label + inf_new_label)
                inf_w.append(2)
    return inf_score, inf_w

def relevance_score(prob_origin_, prob_masked_, labMap, label_gold, label_origin, label_masked):
    '''
    Calculate the relevance score for one perturbation.
    
    Args
        prob_origin_: list of probabilities of model for the original input
        prob_masked_: list of probabilities of model for the perturb input
        lapMap: dictionary of label SRL mapping
        label_gold: list of gold labels 
        label_origin: list of predicted labels for the original input
        label_masked: list of predicted labels for the perturb input
    
    Returns
        rel_score: list of relevance score for each argument change 
    '''
    
    rel_score = []
    rel_w = []
    
    # Judgement space is the set of union of the indices of the argument change
    jud_space = get_idx_arg_preds(label_origin, label_masked, label_gold)
    
    for i in range(len(prob_origin_)):
        if i not in jud_space:
            continue
       
        max_index_origin = np.argmax(prob_origin_[i])
        max_index_masked = np.argmax(prob_masked_[i])
       
        idx_label_gold = labMap[str(label_gold[i])]
        
        # if label gold, label origin and label masked are the same
        if label_gold[i] == label_origin[i] and label_gold[i] == label_masked[i]:
            score_increase_gold = (prob_origin_[i][idx_label_gold] - prob_masked_[i][idx_label_gold])/max(prob_origin_[i][idx_label_gold], prob_masked_[i][idx_label_gold])
            rel_score.append(score_increase_gold)
            rel_w.append(1)
           
        # if label gold and label origin are the same, but label masked is different
        elif label_masked[i] != label_origin[i] and label_origin[i] == label_gold[i]:
            score_increase_gold = (prob_origin_[i][idx_label_gold] - prob_masked_[i][idx_label_gold])/max(prob_origin_[i][idx_label_gold], prob_masked_[i][idx_label_gold])
            score_decrease_mask = (prob_masked_[i][max_index_masked] - prob_origin_[i][max_index_masked])/max(prob_masked_[i][max_index_masked], prob_origin_[i][max_index_masked])
            rel_score.append((score_increase_gold + score_decrease_mask)/2)
            rel_w.append(2)
        
        # if label gold and label masked are the same, but label origin is different   
        elif label_origin[i] != label_masked[i] and label_masked[i] == label_gold[i]:
            score_increase_gold = (prob_origin_[i][idx_label_gold] - prob_masked_[i][idx_label_gold])/max(prob_origin_[i][idx_label_gold], prob_masked_[i][idx_label_gold])
            score_decrease_origin = (prob_masked_[i][max_index_origin] - prob_origin_[i][max_index_origin])/max(prob_masked_[i][max_index_origin], prob_origin_[i][max_index_origin])
            rel_score.append((score_increase_gold + score_decrease_origin)/2)
            rel_w.append(2)
        
        # is label gold, label origin and label masked are different
        elif label_gold[i] != label_origin[i] and label_masked[i] != label_gold[i]:
            score_increase_gold = (prob_origin_[i][idx_label_gold] - prob_masked_[i][idx_label_gold])/max(prob_origin_[i][idx_label_gold], prob_masked_[i][idx_label_gold])
            score_decrease_mask = (prob_masked_[i][max_index_masked] - prob_origin_[i][max_index_masked])/max(prob_masked_[i][max_index_masked], prob_origin_[i][max_index_masked])
            score_decrease_origin = (prob_masked_[i][max_index_origin] - prob_origin_[i][max_index_origin])/max(prob_masked_[i][max_index_origin], prob_origin_[i][max_index_origin])
            rel_score.append((score_increase_gold + score_decrease_mask + score_decrease_origin)/3)
            rel_w.append(1)
           
    return rel_score, rel_w


def brier_score_multi_class(y_true, y_prob, labelMap):
    """
    Calculate the Brier score for multi-class classification using scikit-learn.

    Parameters:
    y_true (numpy.ndarray): True class labels, shape (n_samples,)
    y_prob (numpy.ndarray): Predicted probabilities, shape (n_samples, n_classes)

    Returns:
    float: Brier score
    """
    
    # Predicted probabilities
    y_prob = np.array(y_prob, dtype=float)  
    
    # Ensure y_true is a 1D array
    y_true = np.array(y_true)
    
    # label map 
    y_true = [labelMap[item] for item in y_true]
    y_true_one_hot = label_binarize(y_true, classes=np.arange(y_prob.shape[1]))
  
    # Calculate the Brier score for each class and average them
    brier_scores = np.array([brier_score_loss(y_true_one_hot[:, i], y_prob[:, i], pos_label=1) for i in range(y_prob.shape[1])])
    return round(np.mean(brier_scores), 5) # mean for all classes


def competence_score(pair_score):
    '''
    Calculate the competence score for one perturbation.
    
    Args:
        pair_score: list of dictionaries containing the relevance and influence score for each argument change
    
    Returns:
        comp_score: competence score
        p_value: p-value of the Spearman correlation
        
    '''
    influence_values = [abs(item['influence']) for item in pair_score]
    relevance_values = [item['relevance'] for item in pair_score]

    # Calculate Competence score
    comp_score, p_value = spearmanr(influence_values, relevance_values)
    
    # Get brier score for each unique uid in comp
    brier_score = np.mean([item['brier_score'] for item in pair_score])
    
    return round(comp_score, 4), round(p_value, 4), round(brier_score, 4)
