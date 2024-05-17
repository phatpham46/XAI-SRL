from utils.data_utils import TaskType, NLP_MODELS
from models.model_new import multiTaskModel
from data_preparation import * 
from models.data_manager import allTasksDataset, Batcher, batchUtils
from torch.utils.data import DataLoader
import math
import os
import torch
import pickle as pkl
import logging
        
# def make_masked_data(pred_file_path, out_dir, has_labels, task_name, saved_model_path, type_mask, eval_batch_size=32, max_seq_len=128, seed=42):
#     logger = logging.getLogger("multi_task")
#     device = torch.device('cpu')
#     if torch.cuda.is_available():
#         device = torch.device('cuda')
#     assert os.path.exists(saved_model_path), "saved model not present at {}".format(saved_model_path)
#     assert os.path.exists(pred_file_path), "prediction tsv file not present at {}".format(pred_file_path)
#     loadedDict = torch.load(saved_model_path, map_location=device)
#     taskParamsModel = loadedDict['task_params']
#     logger.info('Task Params loaded from saved model.')
#     assert task_name in taskParamsModel.taskIdNameMap.values(), "task Name not in task names for loaded model"
    
#     taskId = [taskId for taskId, taskName in taskParamsModel.taskIdNameMap.items() if taskName==task_name][0]
#     taskType = taskParamsModel.taskTypeMap[task_name]

#     rows = load_data(pred_file_path, taskType, hasLabels=has_labels)

#     modelName = taskParamsModel.modelType.name.lower()
#     _, _ , tokenizerClass, defaultName = NLP_MODELS[modelName]
#     configName = taskParamsModel.modelConfig
#     if configName is None:
#         configName = defaultName
    
#     tokenizer = tokenizerClass.from_pretrained(configName)
#     logger.info('{} model tokenizer loaded for config {}'.format(modelName, configName))
    
#     dataPath = os.path.join(out_dir, '{}_prediction_data'.format(configName))
#     if not os.path.exists(dataPath):
#         os.makedirs(dataPath)
#     wrtFile = os.path.join(dataPath, '{}.json'.format(pred_file_path.split('/')[-1].split('.')[0]))
#     print('Processing Started...')
#     create_data_ner_new(rows, wrtFile, maxSeqLen=max_seq_len, tokenizer=tokenizer, labelMap=taskParamsModel.labelMap[task_name])
#     print('Data Processing done for {}. File saved at {}'.format(task_name, wrtFile))

#     allTaskslist = [ 
#         {"data_task_id" : int(taskId),
#          "data_path" : wrtFile,
#          "data_task_type" : taskType,
#          "data_task_name" : task_name}
#         ]
#     allData = allTasksDataset(allTaskslist)
#     batchSampler = Batcher(allData, batchSize=eval_batch_size, seed = seed, shuffleBatch=False, shuffleTask=False)
#     batchSamplerUtils = batchUtils(isTrain = False, modelType= taskParamsModel.modelType,
#                                   maxSeqLen = max_seq_len)
#     inferDataLoader = DataLoader(allData, batch_sampler=batchSampler,
#                                 collate_fn=batchSamplerUtils.collate_fn,
#                                 pin_memory=torch.cuda.is_available())

#     allParams = {
#         'task_params': taskParamsModel,
#         'gpu': torch.cuda.is_available(),
#         'num_train_steps': 10,
#         'warmup_steps': 0,
#         'learning_rate': 2e-5,
#         'epsilon': 1e-8
#     }

#     model = multiTaskModel(allParams)
#     model.load_multi_task_model(loadedDict)

#     with torch.no_grad():
#         numTasks = len(taskParamsModel.taskIdNameMap)
#         numStep = math.ceil(len(inferDataLoader)/eval_batch_size)
#         allLabels = [[] for _ in range(numTasks)]
#         allIds = [[] for _ in range(numTasks)]
#         allPreds = [[] for _ in range(numTasks)]
#         allSequenceOutput = [[] for _ in range(numTasks)]
#         allIndexChange = [[] for _ in range(numTasks)]
#         for batchMetaData, batchData in tqdm(inferDataLoader, total=numStep, desc = 'Eval'):
#             batchMetaData, batchData = batchSampler.patch_data(batchMetaData, batchData, gpu=torch.cuda.is_available())
#             prediction, sequenceOutput = model.get_word_embedding(batchMetaData, batchData)
#             batchTaskId = int(batchMetaData['task_id'])
#             orgLabels = batchMetaData['label']
#             allPreds[batchTaskId].extend(prediction)
#             allLabels[batchTaskId].extend(orgLabels)
#             allIds[batchTaskId].extend(batchMetaData['uids'])
#             allSequenceOutput[batchTaskId].extend(sequenceOutput)
            
#         pred_path = pred_file_path.split('/')[-1].split('.')[0]
#         savePath = os.path.join(out_dir, "{}_{}_{}.pkl".format(pred_path, 'masked', type_mask))
#         for i in tqdm(range(len(allPreds)), desc='Saving'):
#             if allPreds[i] == []:
#                 continue
#             taskName = taskParamsModel.taskIdNameMap[i]
#             taskType = taskParamsModel.taskTypeMap[taskName]
#             labMap = taskParamsModel.labelMap[taskName]

#             if taskType == TaskType.NER:
#                 newLabels = []
#                 newIndexSequenceOutput = []
#                 labMapRevN = {v:k for k,v in labMap.items()}

#                 for j, (p, l) in enumerate(zip(allPreds[i], allLabels[i])):
#                     allLabels[i][j] = l[:len(p)]
#                     allPreds[i][j] = [labMapRevN[int(ele)] for ele in p]
#                     allLabels[i][j] = [labMapRevN[int(ele)] for ele in allLabels[i][j]]
                    
#                 newPreds = []
#                 newLabels = []
#                 for m, samp in enumerate(allLabels[i]):
#                     Preds = []
#                     Labels = []
#                     IndexSequenceOutput = []
#                     for n, ele in enumerate(samp):
#                         if ele != '[CLS]' and ele != '[SEP]' and ele != 'X':
#                             Labels.append(ele)
#                             Preds.append(allPreds[i][m][n])
#                             IndexSequenceOutput.append(n)
#                     newLabels.append(Labels)
#                     newPreds.append(Preds)
#                     newIndexSequenceOutput.append(IndexSequenceOutput)
                
#                 allLabels[i] = newLabels
#                 allPreds[i] = newPreds
#                 allIndexChange[i] = newIndexSequenceOutput
#         return allIds[0], allLabels[0], allPreds[0], allIndexChange[0], allSequenceOutput[0]    

class WordEmbeddingAssigner:
    def __init__(self, pred_file_path, out_dir, has_labels, task_name, saved_model_path, masked_data=None, eval_batch_size=32, max_seq_len=128, seed=42):
        self.pred_file_path = pred_file_path
        self.out_dir = out_dir
        self.has_labels = has_labels
        self.task_name = task_name
        self.masked_data = masked_data
        self.saved_model_path = saved_model_path
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.logger = logging.getLogger("multi_task")
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')
        assert os.path.exists(self.saved_model_path), "saved model not present at {}".format(self.saved_model_path)
        assert os.path.exists(self.pred_file_path), "prediction tsv file not present at {}".format(self.pred_file_path)
        loadedDict = torch.load(self.saved_model_path, map_location=device)
        self.taskParamsModel = loadedDict['task_params']
        
        self.taskId = [taskId for taskId, taskName in self.taskParamsModel.taskIdNameMap.items() if taskName==self.task_name][0]
        self.taskType = self.taskParamsModel.taskTypeMap[self.task_name]
        assert self.task_name in self.taskParamsModel.taskIdNameMap.values(), "task Name not in task names for loaded model"
        
        modelName = self.taskParamsModel.modelType.name.lower()
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
        self.logger.info('Task Params loaded from saved model.')
        self.model = multiTaskModel(allParams)
        self.model.load_multi_task_model(loadedDict)
    
    def load_and_create_data(self, isMasked=False, masked_data_type=''):
        self.isMasked = isMasked
        read_file_path = self.pred_file_path if not isMasked else self.pred_file_path.replace('.tsv', f'_masked_{masked_data_type}.tsv')

        rows = load_data(read_file_path, self.taskType, hasLabels=self.has_labels)
        print('Processing Started...')
        dataPath = os.path.join(self.out_dir, '{}_prediction_data'.format(self.configName))
        if not os.path.exists(dataPath):
            os.makedirs(dataPath)
        wrtFile = os.path.join(dataPath, '{}.json'.format(self.pred_file_path.split('/')[-1].split('.')[0]))
        create_data_ner_new(rows, wrtFile, maxSeqLen=self.max_seq_len, tokenizer=self.tokenizer, labelMap=self.taskParamsModel.labelMap[self.task_name])
        print('Data Processing done for {}. File saved at {}'.format(self.task_name, wrtFile))
        self.allTaskslist = [{
            "data_task_id" : int(self.taskId),
            "data_path" : wrtFile,
            "data_task_type" : self.taskType,
            "data_task_name" : self.task_name
        }]
        allData = allTasksDataset(self.allTaskslist)
        self.batchSampler = Batcher(allData, batchSize=self.eval_batch_size, 
                                    seed = self.seed, shuffleBatch=False, shuffleTask=False)
        batchSamplerUtils = batchUtils(isTrain = False, 
                                    modelType= self.taskParamsModel.modelType,
                                    maxSeqLen = self.max_seq_len)
        self.inferDataLoader = DataLoader(allData, batch_sampler=self.batchSampler,
                                    collate_fn=batchSamplerUtils.collate_fn,
                                    pin_memory=torch.cuda.is_available())
        # return self

    def assign_word_embedding_data(self):
        with torch.no_grad():
            numTasks = len(self.taskParamsModel.taskIdNameMap)
            numStep = math.ceil(len(self.inferDataLoader)/self.eval_batch_size)
            
            allLabels = [[] for _ in range(numTasks)]
            allIds = [[] for _ in range(numTasks)]
            allPreds = [[] for _ in range(numTasks)]
            allScores = [[] for _ in range(numTasks)]
            allLogitsSoftmax = [[] for _ in range(numTasks)]
            allSequenceOutput = [[] for _ in range(numTasks)]
            allIndexChangeSequenceOutput = [[] for _ in range(numTasks)]
            
            start_index = 0
            for step, (batchMetaData, batchData) in enumerate(tqdm(self.inferDataLoader, total=numStep, desc='Eval')):
                batchMetaData, batchData = self.batchSampler.patch_data(batchMetaData, batchData, gpu=torch.cuda.is_available())
                
                print('list_uids:', batchMetaData['uids'])
                
                start_index = step * self.eval_batch_size
                end_index = start_index + len(batchMetaData['uids'])
                batchSequenceEmbeddingMasked=None
                if self.masked_data is not None and self.isMasked:
                    batchSequenceEmbeddingMasked = self.masked_data[start_index:end_index]
                prediction, scores, _ouLogitsSoftmax, sequenceOutput = self.model.predict_step(batchMetaData, batchData, sequenceOutputEmbedding=batchSequenceEmbeddingMasked)
                
                batchTaskId = int(batchMetaData['task_id'])
                orgLabels = batchMetaData['label']
                
                allLabels[batchTaskId].extend(orgLabels)
                allPreds[batchTaskId].extend(prediction)
                allScores[batchTaskId].extend(scores)
                allIds[batchTaskId].extend(batchMetaData['uids'])
                allLogitsSoftmax[batchTaskId].extend(_ouLogitsSoftmax)
                allSequenceOutput[batchTaskId].extend(sequenceOutput)
                
            for i in range(len(allPreds)):
                if allPreds[i] == []:
                    continue
                labMap = self.taskParamsModel.labelMap[self.task_name]

                if self.taskType == TaskType.NER:
                    newPreds = []
                    newLabels = []
                    newScores = []
                    newLogitsSigmoid = []
                    newSequenceOutput = []
                    newIndexSequenceOutput = []
                    labMapRevN = {v:k for k,v in labMap.items()}

                    for j, (p, l) in enumerate(zip(allPreds[i], allLabels[i])):
                        allLabels[i][j] = l[:len(p)]
                        allPreds[i][j] = [labMapRevN[int(ele)] for ele in p]
                        allLabels[i][j] = [labMapRevN[int(ele)] for ele in allLabels[i][j]]

                    for m, samp in enumerate(allLabels[i]):
                        Preds = []
                        Labels = []
                        Scores = []
                        LogitsSigmoid = []
                        SeuqenceOutput = []
                        IndexSequenceOutput = []
                        for n, ele in enumerate(samp):
                            if ele != '[CLS]' and ele != '[SEP]' and ele != 'X':
                                Preds.append(allPreds[i][m][n])
                                Labels.append(ele)
                                Scores.append(allScores[i][m][n])
                                LogitsSigmoid.append(allLogitsSoftmax[i][m][n])
                                SeuqenceOutput.append(allSequenceOutput[i][m][n])
                                IndexSequenceOutput.append(n)
                        newPreds.append(Preds)
                        newLabels.append(Labels)
                        newScores.append(Scores)
                        newLogitsSigmoid.append(LogitsSigmoid)
                        newSequenceOutput.append(SeuqenceOutput)
                        newIndexSequenceOutput.append(IndexSequenceOutput)
                    
                    allLabels[i] = newLabels
                    allPreds[i] = newPreds
                    allScores[i] = newScores
                    allLogitsSoftmax[i] = newLogitsSigmoid
                    allSequenceOutput[i] = newSequenceOutput
                    allIndexChangeSequenceOutput[i] = newIndexSequenceOutput
        return allIds[0], allLabels[0], allPreds[0], allScores[0], allLogitsSoftmax[0], allSequenceOutput[0], allIndexChangeSequenceOutput[0]
    
    def get_labelMap(self):
        labelMap = self.taskParamsModel.labelMap[self.task_name]
        labMapRevN = {v:k for k,v in labelMap.items()}
        return labMapRevN