
import os
import numpy as np
import torch

from MLM.mlm_utils.model_utils import NLP, TOKENIZER, MAX_SEQ_LEN, POS_TAG_MAPPING

def check_data_dir(data_dir: str, auto_create=False) -> None:
    """Check if the data directory exists. If it does not exist, create it if auto_create is True.

    Args:
        data_dir (str): Path to the data directory.
        auto_create (bool, optional): auto create or not . Defaults to False.

    Raises:
        FileNotFoundError: 
    """
    
    if not os.path.exists(data_dir):
        if auto_create:
            os.makedirs(data_dir)
        else:
            raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

def get_files(dir: str) -> list:
    '''Function to get the list of files in a given folder'''
    
    files = []
    for path in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, path)):
            files.append(path)
    return files


def get_pos_tag_word(word:str, text:str) -> dict:
    '''
    Return dictionary with key is the word and value is the pos tag of the word'''
    doc = NLP(text)
    word_split = word.split()
    pos_tag_dict = {}
    for token in doc:
        if token.text in word_split:
            pos_tag_dict[token.text] = token.pos_
    return pos_tag_dict


def get_word_list(text: str) -> list:
    '''Function to get the list of words from a given sentence using SpaCy'''
    
    doc = NLP(text)
    word_lst = [word.text for word in [token for token in doc]]
    return word_lst


def pos_tag_mapping(pos_tag):
    if pos_tag == "NOUN":
        return 1
    elif pos_tag =="VERB":
        return 2
    elif pos_tag =="ADJ":
        return 3
    elif pos_tag == "ADV":
        return 4
    else:
        return 0

def get_pos_tag_id(word_dict:dict, pos_tag_dict: dict, label_id:list):
    '''
    Function to get the pos tag id from the label id. 
    for example:
        input:  129, 15, 324, 34, 255, 12
        output  1, 1, 1, 2, 3, 4 (NOUN, NOUN, NOUN, VERB, ADJ, ADV)
    '''
    pos_tag_id = torch.full_like(label_id, fill_value=-1)
    
    for key in pos_tag_dict.keys():
        tokens = word_dict.get(key)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokens = tokens.to(device)
        for i in range(len(label_id) - len(tokens) + 1):
            if torch.equal(torch.as_tensor(label_id[i:i+len(tokens)]).clone().detach(), torch.as_tensor(tokens).clone().detach()):
               
                pos_tag_id[i:i+len(tokens)] = POS_TAG_MAPPING[pos_tag_dict.get(key)]
 
    return pos_tag_id

def get_idx_arg_preds(preds_origin, preds_masked, label_origin=None): # label_origin: nhãn gold
    list_idx_arg_change = []
 
    
    for i in range(min(len(preds_masked), len(preds_origin))):
        test = preds_origin[i].startswith('B-A') or preds_origin[i].startswith('I-A') or preds_masked[i].startswith('B-A') or preds_masked[i].startswith('I-A')
        if label_origin:
            assert len(preds_origin) == len(label_origin), 'Length of preds_origin and label_origin must be the same'
            if test or label_origin[i].startswith('B-A') or label_origin[i].startswith('I-A'):
                list_idx_arg_change.append(i)
        else:
            if test:
                list_idx_arg_change.append(i)
   
    return list_idx_arg_change
    
def decode_token(input_ids: list, skip_special_tokens=False) -> str:
    ''' Funciton to decode the token to text
    '''
    
    return TOKENIZER.decode(input_ids, skip_special_tokens=skip_special_tokens, return_offsets_mapping=True)

def encode_text(text: str) -> dict:
    ''' Function to encode the text '''
    return TOKENIZER.encode_plus(
                    text,
                    max_length=MAX_SEQ_LEN,
                    padding='max_length', 
                    truncation=True,  
                    add_special_tokens = True,
                    return_tensors = 'pt',
                    return_attention_mask = True,
                    return_offsets_mapping=True  
                )




    


           
        