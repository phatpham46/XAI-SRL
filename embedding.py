
import json
import torch
import torch.nn.functional as F
import argparse
import os
import pickle
from models import model
from utils import transform_utils, data_utils 
from transformers import BertTokenizer, BertModel
from models.model import multiTaskModel
from tqdm import tqdm
from torchsummary import summary
bertmodel = BertModel.from_pretrained('dmis-lab/biobert-base-cased-v1.2', output_hidden_states =True)

def cosine_module_2_tensors(tensor1, tensor2):
    
    # Compute the dot product
    dot_product = torch.dot(tensor1, tensor2)

    # Compute the L2 (Euclidean) norms
    norm_tensor1 = torch.norm(tensor1)
    norm_tensor2 = torch.norm(tensor2)

    # Calculate the cosine similarity
    consine = dot_product / (norm_tensor1 * norm_tensor2)
   
    module = 1 - (torch.abs(norm_tensor1 - norm_tensor2)/(norm_tensor1 + norm_tensor2))
   
    cosine_module = 1.0/2 * (module + 1.0/2 * (1 + consine))

    return cosine_module

def cosine_module_similarity(sen1, sen2):
    assert len(sen1) == len(sen2), "Sentence lengths are not equal"
        
    res_vec = []
    for word in range(len(sen1)):
        res_vec.append(cosine_module_2_tensors(sen1[word], sen2[word]))
    return res_vec

def read_data(readPath):
    
    with open(readPath, 'r', encoding = 'utf-8') as file:
        taskData = []
        for i, line in enumerate(file):
            sample = json.loads(line)
            taskData.append(sample)
            
    return taskData

def get_embedding(line):
    
    tokens_id = line['token_id']
    segments_id = line['type_id']
    
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([tokens_id])
    segments_tensors = torch.tensor([segments_id])
    
    with torch.no_grad():
        outputs = bertmodel(tokens_tensor, segments_tensors)
        hidden_states = outputs[2]
        last_hidden_states = outputs[0]
 
    # sum of last four layer
    word_embed_5 = torch.stack(hidden_states[-4:]).sum(0)
    
    # concatenate last four layers
    word_embed_6 = torch.cat([hidden_states[i] for i in [-1,-2,-3,-4]], dim=-1)
    
    return last_hidden_states, word_embed_5, word_embed_6
        
            
def get_embedding_finetuned(line):
       
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Load finetuned model 
    loadedDict = torch.load('./output/multi_task_model_8_11745 (1).pt', map_location=torch.device(device))

    taskParams = loadedDict['task_params']

    allParams = {}
    allParams['task_params'] = taskParams
    allParams['gpu'] = torch.cuda.is_available()
    # dummy values
    allParams['num_train_steps'] = 10
    allParams['warmup_steps'] = 0
    allParams['learning_rate'] = 2e-5
    allParams['epsilon'] = 1e-8

    model = multiTaskModel(allParams)
    model.load_multi_task_model(loadedDict)
    
    tokens_id = line['token_id']
    segments_id = line['type_id']
    
    attention_mask = torch.tensor([line['mask']], device=device)
    
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([tokens_id],  device=device)
    segments_tensors = torch.tensor([segments_id],  device=device)
    
    with torch.no_grad():
        outputs = model.network(tokens_tensor, segments_tensors, attention_mask, 0, 'conllsrl')
       
        hidden_states = outputs[0].hidden_states
    
    ## WORD EMBEDDING
    token_embeddings = torch.stack(hidden_states, dim=0)
    token_embeddings = torch.squeeze(token_embeddings, dim=1)
    token_embeddings = token_embeddings.permute(1,0,2)

    # Stores the token vectors, with shape [22 x 3,072]
    token_vecs_cat = []

    # For each token in the sentence...
    for token in token_embeddings:
        
        cat_vec = torch.cat((token[-1], token[-2], token[-3], token[-4]), dim=0)
        # print("cat_vec shape: ", cat_vec.shape) # 3072
        # Use `cat_vec` to represent `token`.
        token_vecs_cat.append(token[-1])

    #print ('Shape token_vecs_catis: %d x %d' % (len(token_vecs_cat), len(token_vecs_cat[0])))  # 50 x 3072
    return token_vecs_cat
       
def main():
    # parser = argparse.ArgumentParser()
    
    # parser.add_argument('--data_dir', type=str, required=True)
    
    # parser.add_argument('--transform_file', type=str, required=True, default="embedding.yml")
    
    # args = parser.parse_args()
    # transformParams = transform_utils.TransformParams(args.transform_file)
    
    # for transformName, transformFn in transformParams.transformFnMap.items():
    #     transformParameters = transformParams.transformParamsMap[transformName]
    # dataDir = transformParams.readDirMap[transformName]

    # assert os.path.exists(dataDir), "{} doesnt exist".format(dataDir)
    # saveDir = transformParams.saveDirMap[transformName]
    # if not os.path.exists(saveDir):
    #     os.makedirs(saveDir)
 
    # for file in transformParams.readFileNamesMap[transformName]:
    #     #calling respective transform function over file
    #     data_utils.TRANSFORM_FUNCS[transformFn](dataDir = dataDir, readFile=file,
    #                                 wrtDir=saveDir)
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--data_dir', type=str, required=True)
    
    parser.add_argument('--wrt_dir', type=str, required=True)
    
    
    args = parser.parse_args()
    
    if not os.path.exists(args.wrt_dir):
        os.makedirs(args.wrt_dir)
        
    files = []
    for path in os.listdir(args.data_dir):
        if os.path.isfile(os.path.join(args.data_dir, path)):
            files.append(path)
            
   
    for file in files:
        features = {}
        data = read_data(os.path.join(args.data_dir, file))
        print("Reading {}...".format(file))
    
        
        for i, line in tqdm(enumerate(data)):   
            last_hidden_states_original, sum_4_origin, concat_4_origin = get_embedding(line)
           
            last_hidden_states_finetuned, sum_4_finetuned, concat_4_finetuned = get_embedding_finetuned(line)
            
            cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            cosine = cos(last_hidden_states_finetuned[0], last_hidden_states_original[0])
            
            cosine_module = cosine_module_similarity(last_hidden_states_finetuned[0], last_hidden_states_original[0])
        
            feature = {'uid': line['uid'], 'cosine': cosine, 'cosine_module': cosine_module}
            features[i] = feature
            print("feature: ", feature)
            if i % 100 == 0:
                print("done {} rows...".format(i))
        
        with open(os.path.join(args.wrt_dir, 'vecs_{}.pkl'.format(file.split('.')[0])), 'wb') as vecs_wri:
            pickle.dump(features, vecs_wri)
    
 
if __name__ == '__main__':
    main()