
import os
import spacy
import torch
import numpy as np
nlp = spacy.load("en_core_web_sm")
    

def check_data_dir(data_dir, auto_create=False):
    if not os.path.exists(data_dir):
        if auto_create:
            os.makedirs(data_dir)
        else:
            raise FileNotFoundError(f"Data directory {data_dir} does not exist.")

def get_pos_tag_word(word, text):
    doc = nlp(text)
    word_split = word.split()
    pos_tag_dict = {}
    for token in doc:
        if token.text in word_split:
            pos_tag_dict[token.text] = token.pos_
    return pos_tag_dict


def compare_tensors(list1, list2):
    # Check if the lengths of the lists are the same
    if len(list1) != len(list2):
        return False
    
    # Check if each tensor in list1 is equal to the corresponding tensor in list2
    for tensor1, tensor2 in zip(list1, list2):
        if not np.array_equal(tensor1, tensor2):
            return False
    
    return True

def get_key(dictionary, value):
    for key, val in dictionary.items():
        if compare_tensors(val, value):
            return key
    return None  # If value is not found in the dictionary

def get_max_length_word(pred_list):
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
def get_heuristic_word(pos_tag_dict):
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
    
    