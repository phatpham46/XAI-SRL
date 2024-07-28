import joblib
import os
import csv
import json
import pickle
import spacy
import torch
from SRL import model
from transformers import BertModel

SEED = 42

def get_word_list(text: str):
    
    """
    Function to get the list of words from a given sentence using SpaCy.
    
    """
    
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    word_lst = [word.text for word in [token for token in doc]]
   
    return word_lst

def convert_csv_to_txt(dataDir:str, wrtDir:str, readFile:str, transParamDict:dict, isTrainFile=False):
    
    """
    Convert csv with text and arguments to SRL format and save to txt file, 
    each line in the txt file is a word with BIO tag and SRL tag.
    
    """
    
    csv_file = os.path.join(dataDir, readFile)
    txt_file = os.path.join(wrtDir, 'srl_format_{}.txt'.format(readFile.split('.')[0]))
    predicate = readFile.split('_')[0]
    
    with open(csv_file, 'r', encoding='utf-8') as csvfile, open(txt_file, 'w', encoding='utf-8') as txtfile:
        csvreader = csv.DictReader(csvfile)
        for row in csvreader:
            text = row['text'].lower()
            arguments = [(key, get_word_list(value)) for key, value in eval(row['arguments'].lower()).items()]
            tokens = list(map(lambda x: x.lower(), get_word_list(text)))
            arg_list = [None for _ in range(len(tokens))]
            idx_token = 0
            
            while idx_token < len(tokens):
                for tuple_arg in arguments:
                    key, value = tuple_arg
                    
                    if (predicate in tokens[idx_token] or predicate[:-1] in tokens[idx_token]) and arg_list[idx_token] is None:
                        arg_list[idx_token] = 'B-V'
                    
                    if tokens[idx_token] in value[0] and arg_list[idx_token] is None:
                        if tokens[idx_token:idx_token+len(value)] == value:
                          arg_list[idx_token] = 'B-A' + key
                          arg_list[idx_token + 1: idx_token + len(value)] = ['I-A' + key] * (len(value) - 1)
                          idx_token += len(value) - 1
                          break
                
                idx_token += 1
            
            result_list = [item if item is not None else 'O' for item in arg_list]
            for token, arg in zip(tokens, result_list):
                txtfile.write(token + ' B-O ' + arg + '\n')
            txtfile.write('\n')
    print("Done file", readFile)

def pasbio_srl_to_tsv(dataDir:str, readFile:str, wrtDir:str, transParamDict:dict, isTrainFile=False):
    
    """
    This function transforms the data present in coNLL_data/. 
    Raw data is in BIO tagged format with the POS and SRL tags separated by space.
    The transformation function converts the each raw data file into tsv file,
    for SRL task. Following transformed files are written at wrtDir

    - SRL transformed tsv file.
    - SRL label map joblib file.

    For using this transform function, set ``transform_func`` : **snips_intent_ner_to_tsv** in transform file.

    Args:
        dataDir (:obj:`str`) : Path to the directory where the raw data files to be read are present..
        readFile (:obj:`str`) : This is the file which is currently being read and transformed by the function.
        wrtDir (:obj:`str`) : Path to the directory where to save the transformed tsv files.
        transParamDict (:obj:`dict`, defaults to :obj:`None`): Dictionary of function specific parameters. Not required for this transformation function.

    """

    f = open(os.path.join(dataDir, readFile))

    srlW = open(os.path.join(wrtDir, 'srl_{}.tsv'.format(readFile.split('.')[0])), 'w')

    labelMapSrl = {}
    
    sentence = []
    senLens = []
    labelSrl = []
    uid = 0
    print("Making data from file {} ...".format(readFile))
    for i, line in enumerate(f):
        if i%5000 == 0:
            print("Processing {} rows...".format(i))

        line = line.strip(' ') #don't use strip empty as it also removes \n
        wordSplit = line.rstrip('\n').split(' ')
        if len(line)==0 or line.startswith('-DOCSTART') or line[0]=="\n":
            if len(sentence) > 0:
                srlW.write("{}\t{}\t{}\n".format(uid, labelSrl, sentence))
                senLens.append(len(sentence))
                sentence = []
                labelSrl = []
                uid += 1
            continue
            
        sentence.append(wordSplit[0])
        labelSrl.append(wordSplit[-1])
        if isTrainFile:
            if wordSplit[-1] not in labelMapSrl:
                # ONLY TRAIN FILE SHOULD BE USED TO CREATE LABEL MAP FILE.
                labelMapSrl[wordSplit[-1]] = len(labelMapSrl)
          
    print("SRL File Written at {}".format(wrtDir))
  
    # writing label map
    if labelMapSrl != {} and isTrainFile:
        labelMapSrlPath = os.path.join(wrtDir, "srl_{}_label_map.joblib".format(readFile.split('.')[0]))
        joblib.dump(labelMapSrl, labelMapSrlPath)
        print("label Map SRL written at {}".format(labelMapSrlPath))

    f.close()
    srlW.close()
    


            


    