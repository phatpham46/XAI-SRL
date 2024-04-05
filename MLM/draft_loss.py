import torch
import torch.nn as nn
import torch.nn.functional as F
from mlm_utils.preprocess_functions import get_pos_tag_word, get_pos_tag_id, generate_batches
from prepared_for_mlm import get_word_list, decode_token
from mlm_utils.model_utils import BATCH_SIZE, EPOCHS, BIOBERT_MODEL, BERT_PRETRAIN_MODEL, TOKENIZER

def is_POS_match(b_input_id, b_logit_id, b_label_id):
        '''
        Function to check if the POS tag of the masked token in the logits is the same as the POS tag of the masked token in the original text.
        Note: This function assumes that the logits are of shape # ([85, 28996]) 
        lm_label_ids: shape (batch_size, sequence_length)
        '''
        '''cho 1 batch'''
        
        b_matching_term = []
        for idx_sample in range(b_input_id.shape[0]):
            
            input_id = b_input_id[idx_sample]
            logit_id = b_logit_id[idx_sample]
            label_id = b_label_id[idx_sample]
            
            pred_id = input_id.clone() 
            origin_input_id = input_id.clone()
            
            # Find the index of the masked token from lm_label_ids
            mask_index = torch.where(label_id != -100)[0]
            masked_idx_input = torch.where(input_id == TOKENIZER.mask_token_id)[0]
            
            # make sure masked_idx_input and mask_index are the same using asser
            assert torch.equal(mask_index, masked_idx_input), "Masked index and label index are not the same."
            origin_input_id[masked_idx_input] = label_id[mask_index] 
            
            
            # "================= ORIGINAL ============= ")
            # get pos tag of origin text
            origin_text = decode_token(origin_input_id, skip_special_tokens=True) 
            
            # Get masked word in the sentence
            masked_word = decode_token(origin_input_id[mask_index])
            
            pos_tag_origin = get_pos_tag_word(masked_word, origin_text)
            
            word_list = get_word_list(origin_text)
            word_dict = {i: torch.tensor(TOKENIZER.encode_plus(i,add_special_tokens = False)['input_ids'], dtype=torch.int64) for i in word_list}

            pos_tag_id_origin = get_pos_tag_id(word_dict, pos_tag_origin, label_id)
            
            
            # "-============== PREDICTION =================="
            pred = [torch.argmax(logit_id[i]).item() for i in mask_index]

            # Replace the index of the masked token with the list of predicted tokens
            for i in mask_index:
                pred_id[i] = pred[i - mask_index[0]]
            
            
            pos_tag_dict_pred = get_pos_tag_word(decode_token(pred), decode_token(pred_id, skip_special_tokens=True) )
            
            pred_sentence = decode_token(pred_id, skip_special_tokens=True)
        
            pred_word_list = get_word_list(pred_sentence)
            
            word_dict_pred = {i: torch.tensor(TOKENIZER.encode_plus(i, add_special_tokens = False)['input_ids'], dtype=torch.int64) for i in pred_word_list}
        
            # get pos tag for all tokens of each word
            pos_tag_id_pred = get_pos_tag_id(word_dict_pred, pos_tag_dict_pred, pred_id)
        
            matching_term_tensor = torch.zeros_like(pos_tag_id_pred)
            
            matching_term_tensor[mask_index] = torch.where(pos_tag_id_pred[mask_index] == pos_tag_id_origin[mask_index], 
                                        torch.tensor(1), 
                                        torch.tensor(0))

            b_matching_term.append(matching_term_tensor)
            
        return b_matching_term
class CustomLoss(nn.modules.loss._Loss):
    def __init__(self, **kwargs):
        super(CustomLoss, self).__init__(**kwargs)
   
    def forward(self, b_logit_id, b_input_id, b_label_id):
        
        # Cross-entropy term
        b_cross_entropy_term = F.cross_entropy((1-b_logit_id).view(-1, TOKENIZER.vocab_size), b_label_id.view(-1), reduction='none')
    
        # Custom matching term
        b_matching_term = torch.stack(is_POS_match(b_input_id, b_logit_id, b_label_id)).view(-1)

        # Combine terms
        b_loss = 0.5 * ((b_cross_entropy_term) + (1 - b_matching_term))
        
        return b_loss.mean()
    
    
  




    





