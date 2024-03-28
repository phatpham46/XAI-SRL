
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
    for token in doc:
        if token.text in word_split:
            return token.pos_
    return None


def compare_tensors(list1, list2):
    # Check if the lengths of the lists are the same
    if len(list1) != len(list2):
        return False
    
    # Check if each tensor in list1 is equal to the corresponding tensor in list2
    for tensor1, tensor2 in zip(list1, list2):
        if not np.array_equal(tensor1, tensor2):
            return False
    
    return True

def get_key(dictionary, value, count_word):
    unique_set = torch.chunk(value, count_word)[0]
    for key, val in dictionary.items():
        if compare_tensors(val, unique_set):
            return key
    return None  # If value is not found in the dictionary