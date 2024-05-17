
import numpy as np
from utils.data_utils import TaskType, NLP_MODELS
from SRL.model import multiTaskModel
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
        
        loadedDict = torch.load(self.saved_model_path, map_location=self.device)
        self.taskParamsModel = loadedDict['task_params']
        
        self.taskId = [taskId for taskId, taskName in self.taskParamsModel.taskIdNameMap.items() if taskName==self.task_name][0]
        self.taskType = self.taskParamsModel.taskTypeMap[self.task_name]
        assert self.task_name in self.taskParamsModel.taskIdNameMap.values(), "task Name not in task names for loaded model"
        
        modelName = self.taskParamsModel.modelType.name.lower()
        print("Model Name: ", modelName)
        _, _ , tokenizerClass, defaultName = NLP_MODELS[modelName]
        self.configName = self.taskParamsModel.modelConfig
        if self.configName is None:
            self.configName = defaultName
        
        self.tokenizer = tokenizerClass.from_pretrained(self.configName)
        self.logger.info('{} model tokenizer loaded for config {}'.format(modelName, self.configName))

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
        
        
    def load_dataloader(self):
        self.dataset = PerturedDataset(self.data_file, self.device)
        self.dataloader = self.dataset.generate_batches(self.dataset, self.eval_batch_size)
        
    
    def get_predictions(self, is_masked=False):
        self.model.eval()
        
        allPreds = []
        allScores = []
        allLogitsSoftmax = []
        allLogitsRaw = []
        allLabels = []
        for batch in tqdm(self.dataloader, total = len(self.dataloader)):
            batch = tuple(t.to(self.device) for t in batch)

            if is_masked: # file masked_data_json
                origin_uid, origin_id, _, token_id, type_id, mask = batch
            else: # file mlm_output
                origin_uid, origin_id, token_id, mask, type_id, pos_tag_id = batch 
                
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
            
        # flatten allPreds, allScores
        allPreds = [item for sublist in allPreds for item in sublist]
        allScores = [item for sublist in allScores for item in sublist]
        allLogitsSoftmax = [item for sublist in allLogitsSoftmax for item in sublist]
        allLogitsRaw = [item for sublist in allLogitsRaw for item in sublist]
        return allPreds, allScores, allLogitsSoftmax, allLogitsRaw     