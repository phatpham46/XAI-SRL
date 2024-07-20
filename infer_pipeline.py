"""
Pipeline for inference on batch for multi-task
"""
from MLM.mlm_utils.pertured_dataset import PerturbedDataset
from utils.data_utils import TaskType, NLP_MODELS
from SRL.model import multiTaskModel
from transformers import BertModel, BertTokenizer
from data_preparation import * 
import os
import torch
class inferPipeline:

    """
    For running inference on samples using a trained model,

    Args:
        modelPath (:obj:`str`) : Path to the trained model.
        maxSeqLen (:obj:`int`, defaults to :obj:`128`) : maximum sequence length to be considered for samples.
        Truncating and padding will happen accordingly.
        
    Example::

        >>> from infer_pipeline import inferPipeline
        >>> pipe = inferPipeline(modelPath = 'sample_out_dir/multi_task_model.pt', maxSeqLen = 50)
    
    """

    def __init__(self, logger, modelPath=None, maxSeqLen = 85):

        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda')

        self.maxSeqLen = maxSeqLen
        self.modelPath = modelPath
        
        if self.modelPath is not None and os.path.exists(self.modelPath):

            loadedDict = torch.load(self.modelPath, map_location=device)
            self.taskParams = loadedDict['task_params']
            logger.info('Task Params loaded from saved model.')

            modelName = self.taskParams.modelType.name.lower()
            _, _ , tokenizerClass, defaultName = NLP_MODELS[modelName]
            configName = self.taskParams.modelConfig
            if configName is None:
                configName = defaultName
            #making tokenizer for model
            self.tokenizer = tokenizerClass.from_pretrained(configName)
            
            logger.info('{} model tokenizer loaded for config {}'.format(modelName, configName))
        
            allParams = {}
            allParams['task_params'] = self.taskParams
            allParams['gpu'] = torch.cuda.is_available()
           
            # dummy values
            allParams['num_train_steps'] = 10
            allParams['warmup_steps'] = 0
            allParams['learning_rate'] = 2e-5
            allParams['epsilon'] = 1e-8

            #making and loading model
            self.model = multiTaskModel(allParams)
            self.model.load_multi_task_model(loadedDict)
       
        else:
            self.model = BertModel.from_pretrained('dmis-lab/biobert-base-cased-v1.2', output_hidden_states =True)
            self.tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
            logger.info('BioBERT model loaded.')
       
    def infer(self, dataDir, file, batch_size = 32, device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") ):
        dataset = PerturbedDataset(
                        file_name=dataDir/file,
                        device = device)

        dataloader = dataset.generate_batches(
                        dataset= dataset,
                        batch_size=batch_size)
        self.dataloader = dataloader