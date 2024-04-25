
import os
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from mlm_utils.model_utils import NLP, TOKENIZER, MAX_SEQ_LEN, POS_TAG_MAPPING

def check_data_dir(data_dir: str, auto_create=False) -> None:
    """ Check if the data directory exists. If it does not exist, create it if auto_create is True.

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
    '''Function to get the list of words from a given text using spacy'''
    
    doc = NLP(text)
    word_lst = [word.text for word in [token for token in doc]]
    return word_lst


def compare_tensors(list1: torch.Tensor, list2: torch.Tensor) -> bool:
    '''Function to compare two lists of tensors '''

    
    # Check if the lengths of the lists are the same
    if len(list1) != len(list2):
        return False
    
    # Check if each tensor in list1 is equal to the corresponding tensor in list2
    for tensor1, tensor2 in zip(list1, list2):
        if not np.array_equal(tensor1, tensor2):
            return False
    
    return True

def get_key(dictionary: dict, value: torch.Tensor) -> str:
    ''' Function to get the key from a dictionary given a value '''
    for key, val in dictionary.items():
        if compare_tensors(val, value):
            return key
    return None  # If value is not found in the dictionary


def get_max_length_word(pred_list: list) -> int:
    '''Function to get the longest length word
    Input: 
        pred_list: ["Ba", "masture", "ofs", "a"]
    Output:
        max_length_word: "masture"
    '''
    max_len = 0
    for pred in pred_list:
        if len(pred) > max_len:
            max_len = len(pred)
    
    return max_len

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
               
                # pos_tag_id[i:i+len(tokens)] = pos_tag_mapping(pos_tag_dict.get(key))
                pos_tag_id[i:i+len(tokens)] = POS_TAG_MAPPING[pos_tag_dict.get(key)]
 
    return pos_tag_id

def get_heuristic_word(pos_tag_dict: dict) -> tuple:
    '''Function to get the longest length word, if it has more than 2 words, will return the content word
    Input: 
        pred_list: ["Ba", "masture", "ofs", "a"]
        pos_tag_list: ["ADJ", "NOUN", "NOUN", "NOUN"]
    Output:
        heuristic_word: "masture"
    '''
    max_len = get_max_length_word(pos_tag_dict.keys())
    heuristic_word = {}
    # get the word having length = maxlen
    for key, value in pos_tag_dict.items():
        if len(key) == max_len:
            heuristic_word[key] = value
    
    if len(heuristic_word) > 1:
        # get the content word
        for word, pos_tag in heuristic_word.items():
            if pos_tag in ["NOUN", "VERB", "ADJ", "ADV"]:
                return word, pos_tag
    return heuristic_word.keys(), heuristic_word.values()      
    
    
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
                    return_tensors="pt",  
                    return_attention_mask = True,
                    return_offsets_mapping=True  
                )


def create_list_content_word(dataDir, wriDir):
    '''
    create_list_content_word("./interim", "./list_content_word" )
    '''
    check_data_dir(dataDir, False)
    check_data_dir(wriDir, True)
    
    lists = {"NOUN": [], "VERB": [], "ADJ": [], "ADV": []}
    for file in get_files(dataDir):
        data = pd.read_csv(os.path.join(dataDir, file))
        print("Processing file:", file)
        
        with tqdm(total=len(data['text'])) as pbar:
            for sample in data['text']: 
                doc = NLP(sample)
                for token in doc:
                    if token.pos_ in lists:
                        lists[token.pos_].append({"word": token.text, "tokens": TOKENIZER.encode(token.text, add_special_tokens=False)})
                pbar.update(1)
        
    for pos, lst in lists.items():
        df = pd.DataFrame(lst)
        df['word'] = df['word'].str.lower()
        df.drop_duplicates(subset='word', keep = 'first', inplace = True)  # Remove duplicates
        df.to_csv(os.path.join(wriDir, f'{pos.lower()}.csv'), index=False)
    
    print("Done!")

           
        