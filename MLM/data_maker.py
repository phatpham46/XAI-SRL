
import numpy as np
import sys
sys.path.append('../')
from SRL.model import multiTaskModel
from utils.data_utils import NLP_MODELS
from data_preparation import * 
from mlm_utils.pertured_dataset import PerturedDataset
import torch.nn as nn
import math
import os
import torch

class DataMaker():
    def __init__(self, data_file, out_dir, saved_model_path, eval_batch_size=32, max_seq_len=85, seed=42):
        self.data_file = data_file
        self.out_dir = out_dir
        self.saved_model_path = saved_model_path
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        assert os.path.exists(self.saved_model_path), "saved model not present at {}".format(self.saved_model_path)
        assert os.path.exists(self.data_file), "prediction tsv file not present at {}".format(self.data_file)
        
        self.dataset = PerturedDataset(self.data_file, self.device)
        self.dataloader = self.dataset.generate_batches(self.dataset, self.eval_batch_size)
        loadedDict = torch.load(self.saved_model_path, map_location=self.device)
        self.taskParamsModel = loadedDict['task_params']
        
        modelName = self.taskParamsModel.modelType.name.lower()
        print("Model Name: ", modelName)
        _, _ , tokenizerClass, defaultName = NLP_MODELS[modelName]
        self.configName = self.taskParamsModel.modelConfig
        if self.configName is None:
            self.configName = defaultName
        
        self.tokenizer = tokenizerClass.from_pretrained(self.configName)
        
        allParams = {
            'task_params': self.taskParamsModel,
            'gpu': torch.cuda.is_available(),
            'num_train_steps': 10,
            'warmup_steps': 0,
            'learning_rate': 2e-5,
            'epsilon': 1e-8
        }
        
        
        self.model = multiTaskModel(allParams)
        self.model.load_multi_task_model(loadedDict)
        
    
    
    def get_predictions(self, is_masked=False):
        self.model.network.eval()
        
        allPreds = []
        allScores = []
        allLogitsSoftmax = []
        allLogitsRaw = []
        allLabels = []
        allOriginUIDs = []
        for batch in tqdm(self.dataloader, total = len(self.dataloader)):
            batch = tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch)

            if is_masked: # file masked_data_json
                origin_uid, origin_id, _, token_id, type_id, mask = batch
            else: # file mlm_output
                origin_uid, token_id, mask_id, mask, type_id, pos_tag_id = batch 
            with torch.no_grad():
                _, logits = self.model.network(token_id, type_id, mask, 0, 'conllsrl')
            
                outLogitsSoftmax = nn.functional.softmax(logits, dim = 2).data.cpu().numpy()
                
                outLogitsSigmoid = nn.functional.sigmoid(logits).data.cpu().numpy()
                
                predicted_sm = np.argmax(outLogitsSoftmax, axis = 2)
                
            
                # here in score, we only want to give out the score of the class of tag, which is maximum
                predScore = np.max(outLogitsSigmoid, axis = 2).tolist() 
                
                predicted_sm = predicted_sm.tolist()
            
                # get the attention masks, we need to discard the predictions made for extra padding
                predictedTags = []
                predScoreTags = []
                
                if mask is not None:
                    #shape of attention Masks (batchSize, maxSeqLen)
                    actualLengths = mask.cpu().numpy().sum(axis = 1).tolist()
                
                    for i, (pred, sc) in enumerate(zip(predicted_sm, predScore)):
                        predictedTags.append( pred[:actualLengths[i]] )
                        predScoreTags.append( sc[:actualLengths[i]])
        
                else:
                    predictedTags = predicted_sm
                    predScoreTags = predScore
                
                allPreds.append(predictedTags)  
                allScores.append(predScoreTags)  
                allLogitsSoftmax.append(outLogitsSoftmax)
                # allLabels.append(label.tolist())
                allLogitsRaw.append(logits.data.cpu().numpy())
                allOriginUIDs.append(origin_uid)
        # flatten allPreds, allScores
        allOriginUIDs = [item for sublist in allOriginUIDs for item in sublist]
        allPreds = [item for sublist in allPreds for item in sublist]
        allScores = [item for sublist in allScores for item in sublist]
        allLogitsSoftmax = [item for sublist in allLogitsSoftmax for item in sublist]
        allLogitsRaw = [item for sublist in allLogitsRaw for item in sublist]
        return allOriginUIDs, allPreds, allScores, allLogitsSoftmax, allLogitsRaw     