
import os
import spacy
import numpy as np
import torch
from mlm_utils.model_utils import BATCH_SIZE, EPOCHS, BIOBERT_MODEL, BERT_PRETRAIN_MODEL, TOKENIZER, NUM_CPU, MAX_SEQ_LEN

from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
nlp = spacy.load("en_core_web_sm")
    

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

def get_sampler(local_rank, dataset):
    if local_rank == -1:
        return RandomSampler(dataset)
    else:
        return SequentialSampler(dataset)
    

def generate_batches(local_rank, dataset, batch_size,
    drop_last=True, device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will
    ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(
        dataset=dataset, 
        sampler=get_sampler(local_rank, dataset),
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=NUM_CPU)  

    return dataloader

def get_pos_tag_word(word, text) :
    
    doc = nlp(text)
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
    if pos_tag =="NOUN":
        return 1
    elif pos_tag =="VERB":
        return 2
    elif pos_tag =="ADJ":
        return 3
    elif pos_tag == "ADV":
        return 4
    else:
        return -1

def get_pos_tag_id(args, word_dict, pos_tag_dict, label_id):
    pos_tag_id = torch.full_like(label_id, fill_value=-1)
    
    for key in pos_tag_dict.keys():
        tokens = word_dict.get(key)
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        tokens = tokens.to(device)
        for i in range(len(label_id) - len(tokens) + 1):
            if torch.equal(torch.as_tensor(label_id[i:i+len(tokens)]).clone().detach(), torch.as_tensor(tokens).clone().detach()):
               
                pos_tag_id[i:i+len(tokens)] = pos_tag_mapping(pos_tag_dict.get(key))
 
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
    
    