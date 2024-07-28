
import json
import os
import sys
sys.path.append('../')
from utils.data_utils import NLP, MAX_SEQ_LEN 

def read_data(readPath):
    """Read json data and return as list of dictionaries.

    Args:
        readPath (`obj`: str): path to the file to read

    Returns:
        list: list of dictionaries 
    """
    
    with open(readPath, 'r', encoding = 'utf-8') as file:
        taskData = list(map(json.loads, file))
          
    return taskData

def check_data_dir(data_dir: str, auto_create=False):
    """
    Check if the data directory exists. If it does not exist, create it if auto_create is True.

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

def get_files(dir: str):
    """
    Function to get the list of files in a given folder.
    
    Args:   
        dir (str): Path to the directory
        
    Returns:
        list: list of files in the directory
    """
    
    files = []
    for path in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, path)):
            files.append(path)
    return files


def get_pos_tag_word(word:str, text:str):
    """
    Return dictionary with key is the word and value is the pos tag of each token for that word.
    
    Args:
        word (str): The word to get the pos tag
        text (str): The text that contains the word

    Returns:
        dict: dictionary with key is the word and value is the pos tag of each token for that word.
    """
    doc = NLP(text)
    word_split = word.split()
    pos_tag_dict = {}
    for token in doc:
        if token.text in word_split:
            pos_tag_dict[token.text] = token.pos_
    return pos_tag_dict


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
    
def decode_token(input_ids: list, tokenizer, skip_special_tokens=False):
    """
    Funciton to decode the token to text.
    
    Args:
        input_ids (list): list of input ids
        tokenizer (obj): tokenizer object
        skip_special_tokens (bool): skip special tokens or not
        
    Returns:
        str: decoded text
    
    """
    
    return tokenizer.decode(input_ids, skip_special_tokens=skip_special_tokens, return_offsets_mapping=True)

def encode_text(text: str, tokenizer):
    """
    Function to encode the text using the tokenizer.
    
    Args: 
        text (str): text to encode
        tokenizer (obj): tokenizer object
        
    Returns:
        dict: dictionary containing the encoded text
    
    """
    
    return tokenizer.encode_plus(
                    text,
                    max_length=MAX_SEQ_LEN,
                    padding='max_length', 
                    truncation=True,  
                    add_special_tokens = True,
                    return_tensors = 'pt',
                    return_attention_mask = True,
                    return_offsets_mapping=True  
                )




    


           
        