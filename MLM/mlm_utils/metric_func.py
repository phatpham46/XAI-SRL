import ast

import numpy as np


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