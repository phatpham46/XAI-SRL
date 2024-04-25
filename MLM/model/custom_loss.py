import torch
import torch.nn as nn
import torch.nn.functional as F
from mlm_utils.transform_func import get_pos_tag_word, get_pos_tag_id
from prepared_for_mlm import get_word_list, decode_token
from mlm_utils.model_utils import TOKENIZER

def get_pos_tag(token_id, origin_input_id, label_id):
    '''
    token_id: id token of masked word (only tokens of one word, not the whole sentence)
    origin_input_id: id token of original sentence(has not been masked yet)
    label_id: 
    '''
    # get pos tag of origin text
    origin_text = decode_token(origin_input_id, skip_special_tokens=True) 
    
    # Get masked word in the sentence
    masked_word = decode_token(token_id)
    
    # Get pos tag of masked word in the sentence
    pos_tag_origin = get_pos_tag_word(masked_word, origin_text)
    
    # Get pos tag id of masked word in the sentence. 
    # For example, if masked word is Noun and it has 2 tokens, then the pos tag id will be [1, 1], the remain token will be 0
    
    word_list = get_word_list(origin_text)
    word_dict = {i: torch.tensor(TOKENIZER.encode_plus(i,add_special_tokens = False)['input_ids'], dtype=torch.int64) for i in word_list}
    pos_tag_id_origin = get_pos_tag_id(word_dict, pos_tag_origin, label_id)
    
    return pos_tag_id_origin

        

class CustomLoss(nn.modules.loss._Loss):
    def __init__(self, **kwargs):
        super(CustomLoss, self).__init__(**kwargs)
   
    def forward(self, b_logit_id, b_input_id, b_label_id):
        
        # Cross-entropy term
        b_cross_entropy_term = F.cross_entropy((1-b_logit_id).view(-1, TOKENIZER.vocab_size), b_label_id.view(-1), reduction='none')
    
        # Custom matching term
        b_matching_term = torch.stack(pos_tag_id_mlm_data(b_input_id, b_logit_id, b_label_id)[0]).view(-1).float().requires_grad_(True)

        # Combine terms
        b_loss = 0.5 * ((b_cross_entropy_term).requires_grad_(True) + (1.0 * (1 - b_matching_term)).requires_grad_(True))
        # b_loss = (1.0 * (1 - b_matching_term)).requires_grad_(True)
        return b_loss.mean()
    
    
  




    





