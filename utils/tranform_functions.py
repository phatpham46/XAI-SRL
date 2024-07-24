import joblib
import os
import csv
import json
import pickle
import torch
from SRL import model
from transformers import BertModel
from mlm_utils.transform_func import get_word_list
SEED = 42


def convert_csv_to_txt(dataDir:str, wrtDir:str, readFile:str, transParamDict:dict, isTrainFile=False) -> None:
    '''
    Convert csv with text and arguments to SRL format and save to txt file, each line in the txt file is a word with BIO tag and SRL tag.
    '''
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

def pasbio_srl_to_tsv(dataDir:str, readFile:str, wrtDir:str, transParamDict:dict, isTrainFile=False) -> None:
    
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
  
    #writing label map
    if labelMapSrl != {} and isTrainFile:
        labelMapSrlPath = os.path.join(wrtDir, "srl_{}_label_map.joblib".format(readFile.split('.')[0]))
        joblib.dump(labelMapSrl, labelMapSrlPath)
        print("label Map SRL written at {}".format(labelMapSrlPath))

        
    f.close()
    srlW.close()
    
bertmodel = BertModel.from_pretrained('dmis-lab/biobert-base-cased-v1.2', output_hidden_states =True)


def read_data(readPath):

    with open(readPath, 'r', encoding = 'utf-8') as file:
        taskData = list(map(json.loads, file))
          
    return taskData


def get_embedding(dataDir, readFile, wrtDir):
    
    data = read_data(os.path.join(dataDir, readFile))
    features = []
    for i, line in enumerate(data):
        tokens_id = line['token_id']
        segments_id = line['type_id']
        u_id = line['uid']
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([tokens_id])
        
        segments_tensors = torch.tensor([segments_id])
        
        print("Processed {} rows...".format(i))
        with torch.no_grad():
            outputs = bertmodel(tokens_tensor, segments_tensors)
            print("")
            hidden_states = outputs[2]
            print("done {} rows...".format(i))
        # `hidden_states` is a Python list.
        # Each layer in the list is a torch tensor.
        # `token_vecs` is a tensor with shape [50 x 768]
        
        ## WORD EMBEDDING
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)

        token_embeddings.size()

        # Stores the token vectors, with shape [22 x 3,072]
        token_vecs_cat = []

        # `token_embeddings` is a [22 x 12 x 768] tensor.

        # For each token in the sentence...
        for token in token_embeddings:
            # `token` is a [12 x 768] tensor

            # Concatenate the vectors (that is, append them together) from the last four layers.
            # Each layer vector is 768 values, so `cat_vec` is length 3,072.
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            
            # Use `cat_vec` to represent `token`.
            token_vecs_cat.append(cat_vec)

        features.append({'uid': u_id, 'vec': token_vecs_cat})
        
    # write u_id and token_vecs_cat to file
    with open(os.path.join(wrtDir, 'vecs_{}.pkl'.format(readFile.split('.')[0])), 'wb') as vecs_wri:
        pickle.dump(features, vecs_wri)
            
def get_embedding_finetuned(dataDir, readFile, wrtDir):
    data = read_data(os.path.join(dataDir, readFile))
    
    # Load finetuned model 
    loadedDict = torch.load('../output/multi_task_model_9_13050.pt', map_location=torch.device('cpu'))

    taskParams = loadedDict['task_params']

    allParams = {}
    allParams['task_params'] = taskParams
    allParams['gpu'] = torch.cuda.is_available()
    # dummy values
    allParams['num_train_steps'] = 10
    allParams['warmup_steps'] = 0
    allParams['learning_rate'] = 2e-5
    allParams['epsilon'] = 1e-8

    multiTask = model.SRLModelTrainer(allParams)
    multiTask.load_multi_task_model(loadedDict)
    
    for i, line in enumerate(data):
        tokens_id = line['token_id']
        segments_id = line['type_id']
        u_id = line['uid']
        attention_mask = line['mask']
        
        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([tokens_id])
        
        segments_tensors = torch.tensor([segments_id])
        
        print("Processed {} rows...".format(i))
        with torch.no_grad():
            outputs = multiTask.network(tokens_tensor, segments_tensors, attention_mask, 0, 'conllsrl')
            print("")
            hidden_states = outputs[1][2]
            print("done {} rows...".format(i))
        # `hidden_states` is a Python list.
        # Each layer in the list is a torch tensor.
        # `token_vecs` is a tensor with shape [50 x 768]
        
        ## WORD EMBEDDING
        token_embeddings = torch.stack(hidden_states, dim=0)
        token_embeddings = torch.squeeze(token_embeddings, dim=1)
        token_embeddings = token_embeddings.permute(1,0,2)

        token_embeddings.size()

        # Stores the token vectors, with shape [22 x 3,072]
        token_vecs_cat = []

        # `token_embeddings` is a [22 x 12 x 768] tensor.

        # For each token in the sentence...
        for token in token_embeddings:
            # `token` is a [12 x 768] tensor

            # Concatenate the vectors (that is, append them together) from the last four layers.
            # Each layer vector is 768 values, so `cat_vec` is length 3,072.
            cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
            
            # Use `cat_vec` to represent `token`.
            token_vecs_cat.append(cat_vec)

        features = {
                'uid': u_id,
                'vec': token_vecs_cat}
        # write u_id and token_vecs_cat to file
        with open(os.path.join(wrtDir, 'finetuned_vecs_{}.pkl'.format(readFile.split('.')[0])), 'wb') as vecs_wri:
            pickle.dump(features, vecs_wri)
            


    