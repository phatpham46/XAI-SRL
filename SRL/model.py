import torch
import torch.nn as nn
import logging
import numpy as np
import SRL.dropout 
from utils import data_utils
from transformers import get_linear_schedule_with_warmup
logger = logging.getLogger("multi_task")

class SharedModelNetwork(nn.Module):
    def __init__(self, params):
        super(SharedModelNetwork, self).__init__()
        self.params = params
        self.taskParams = self.params['task_params']
        assert self.taskParams.modelType in data_utils.ModelType._value2member_map_, "Model Type is recognized, check in data_utils"
        self.modelType = self.taskParams.modelType

        # making shared base encoder model
        # Initializing with a config file does not load the weights associated with the model,
        # only the configuration. Check out the from_pretrained() method to load the model weights.
        
        modelName = self.modelType.name.lower()
        _, modelClass, _, defaultName = data_utils.NLP_MODELS[modelName]
        if self.taskParams.modelConfig is not None:
            logger.info('Making shared model from given config name {}'.format(self.taskParams.modelConfig))
            self.sharedModel = modelClass.from_pretrained(self.taskParams.modelConfig, output_hidden_states=True)
        else:
            logger.info("Making shared model with default config")
            self.sharedModel = modelClass.from_pretrained(defaultName, output_hidden_states=True)
        self.hiddenSize = self.sharedModel.config.hidden_size
        
        #making headers
        self.allDropouts, self.allHeaders = self.make_multitask_heads()
        
        #making pooler layer. Will be used as required
        self.poolerLayer = self.make_pooler_layer()
        self.initialize_headers()

    def make_multitask_heads(self):
        '''
        Function to make task specific headers for all tasks.
        hiddenSize :- Size of the encoder hidden state output from shared model. Hidden state will be 
        used as input to headers. Hence size is required to make input layer for header.

        The final layer output size depends on the number of classes expected in the output
        '''
        allHeaders = nn.ModuleDict()
        allDropouts = nn.ModuleDict()

        # taskIdNameMap is orderedDict, it will preserve the order of tasks
        for taskId, taskName in self.taskParams.taskIdNameMap.items():
            # taskId:0, taskName:conllsrl
            
            taskType = self.taskParams.taskTypeMap[taskName]
            numClasses = int(self.taskParams.classNumMap[taskName])
            dropoutValue = self.taskParams.dropoutProbMap[taskName]
            dropoutLayer = SRL.dropout.DropoutWrapper(dropoutValue)
            outLayer = nn.Linear(self.hiddenSize, numClasses)
            allDropouts[taskName] = dropoutLayer
            allHeaders[taskName] = outLayer
            
        return allDropouts, allHeaders

    def initialize_headers(self):
        def init_weights(module):
            if isinstance(module, (nn.Linear, nn.Embedding)):
                module.weight.data.normal_(mean=0.0, std=0.02 * 1.0)
            if isinstance(module, nn.Linear):
                if module.bias is not None:
                    module.bias.data.zero_()

        self.apply(init_weights)
    
    def make_pooler_layer(self):
        '''
        Function to make pooler output from hidden state output in case pooler output is not returned from 
        shared model forward
        '''
        poolerLayer = nn.Linear(self.hiddenSize, self.hiddenSize)
        
        return poolerLayer

    def forward(self, tokenIds=None, typeIds=None, attentionMasks=None, taskId=0, taskName='conllsrl'):

        # Taking out output from shared encoder. 
        outputs = self.sharedModel(input_ids = tokenIds, token_type_ids = typeIds, attention_mask = attentionMasks)

        # SequenceOutput has shape: (batchSize, maxSeqLen, hiddenSize=768))
        sequenceOutput = self.allDropouts[taskName](outputs[0])
        
        # sequenceOutput = outputs[0]
        logits = self.allHeaders[taskName](sequenceOutput)
        
        return outputs, logits
    
   
class SRLModelTrainer:
    '''
    This is the model helper class which is responsible for building the 
    model architecture and training. It has following functions
    1. Make the shared task network
    2. set optimizer with linear scheduler and warmup
    3. Multi-gpu support
    4. Task specific loss function
    5. Model update for training
    6. Predict function for inference
    '''
    def __init__(self, params):
        self.params = params
        self.taskParams = self.params['task_params']

        self.globalStep = 0
        self.accumulatedStep = 0

        # making model
        if torch.cuda.device_count() > 1:
            logger.info("Using number of gpus: {}".format(torch.cuda.device_count()))
            self.network = nn.DataParallel(SharedModelNetwork(params))
        else:
            self.network = SharedModelNetwork(params)
            logger.info('Using single GPU')


        # transfering to gpu if available
        if self.params['gpu']:
            self.network.cuda()

        
        # optimizer and scheduler
        self.optimizer, self.scheduler = self.make_optimizer(numTrainSteps=self.params['num_train_steps'],
                                                            warmupSteps=self.params['warmup_steps'],
                                                            lr = self.params["learning_rate"],
                                                            eps = self.params["epsilon"])
        # loss class list
        self.lossClassList = self.make_loss_list()


    def make_optimizer(self, numTrainSteps, lr, eps, warmupSteps=0):
        
        # we will use AdamW optimizer from huggingface       
        optimizer = torch.optim.AdamW(self.network.parameters(), lr=lr, eps = eps)

        
        # lr scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmupSteps,
                                                    num_training_steps=numTrainSteps)
       
        logger.debug("optimizer and scheduler created")
        return optimizer, scheduler

    def make_loss_list(self):
        # making loss class list according to task id
        lossClassList = []
        for taskId, taskName in self.taskParams.taskIdNameMap.items():
            lossName = self.taskParams.lossMap[taskName].name.lower()
            lossClass = data_utils.LOSSES[lossName](alpha=self.taskParams.lossWeightMap[taskName])
            lossClassList.append(lossClass)
        return lossClassList

    def _to_cuda(self, tensor):
        if tensor is None: return tensor

        if isinstance(tensor, list) or isinstance(tensor, tuple):
            y = [e.cuda(non_blocking=True) for e in tensor]
            for e in y:
                e.requires_grad = False
        else:
            y = tensor.cuda(non_blocking=True)
            y.requires_grad = False
        return y 

    def update_step(self, batchMetaData, batchData):
        # train mode
        self.network.train()
        target = batchData[batchMetaData['label_pos']]
        
        # send label to gpu if present
        if self.params['gpu']:
            target = self._to_cuda(target)


        taskId = batchMetaData['task_id']
        taskName = self.taskParams.taskIdNameMap[taskId]
        logger.debug('task id for batch {}'.format(taskId))
        
        
        # making forward pass
        # batchData: [tokenIdsBatchTensor, typeIdsBatchTensor, masksBatchTensor, labelsTensor]
        # batchData would have typeIdsBatchTensor or masksBatchTensor as None if the model doesn't support it
        # model forward function input [tokenIdsBatchTensor, typeIdsBatchTensor, masksBatchTensor, taskId]
        # we are not going to send labels in batch
        logger.debug('len of batch data {}'.format(len(batchData)))
        logger.debug('label position in batch data {}'.format(batchMetaData['label_pos']))
        modelInputs = batchData[:batchMetaData['label_pos']]
        
       
        modelInputs += [taskId]
        modelInputs += [taskName]
        
        logger.debug('size of model inputs {}'.format(len(modelInputs)))
        logits = self.network(*modelInputs)[1]
        
        #calculating task loss
        self.taskLoss = 0
        logger.debug('size of model output logits {}'.format(logits.size()))
        logger.debug('size of target {}'.format(target.size())) #torch.Size([32, 50])
        if self.lossClassList[taskId] and (target is not None):
            self.taskLoss = self.lossClassList[taskId](logits, target, attnMasks=modelInputs[2])
           
            # tensorboard details
            self.tbTaskId = taskId
            self.tbTaskLoss = self.taskLoss.item()
        
        taskLoss = self.taskLoss / self.params['grad_accumulation_steps']
        taskLoss.backward()
        self.accumulatedStep += 1

        # gradients will be updated only when accumulated steps become mentioned number in grad_acc_steps (one global update)
        if self.accumulatedStep == self.params['grad_accumulation_steps']:
            logging.debug('model updated.')
            if self.params['grad_clip_value'] > 0:
                torch.nn.utils.clip_grad_norm_(self.network.parameters(),
                                                self.params['grad_clip_value'])

            self.optimizer.step()
            self.scheduler.step()

            self.optimizer.zero_grad()
           
            # reset accumulated steps
            self.accumulatedStep = 0
        
    def predict_step(self, batchMetaData, batchData):
        '''
        Function for predicting on a batch from model. Will be used for inference and dev/test set.
        The labels in case of eval (predictions) are kept in batchMetaData and are not Tensors
        '''
        self.network.eval()
        taskId = batchMetaData['task_id']
        taskName = self.taskParams.taskIdNameMap[taskId]
      
        modelInputs = batchData + [taskId] + [taskName]
        logger.debug("Pred model input length: {}".format(len(modelInputs)))

        #making forward pass to get logits
        outLogits = self.network(*modelInputs)[1]
        logger.debug("Pred model logits shape: {}".format(outLogits.size()))
      
    
        outLogitsSoftmax = nn.functional.softmax(outLogits, dim = 2).data.cpu().numpy()
        #shape of outlogits now (batchSize, maxSeqLen, classNum)
        
        predicted = np.argmax(outLogitsSoftmax, axis = 2)
        
        #shape of predicted now (batchSize, maxSeqLen)
        logger.debug("Final Predictions shape after argmx: {}".format(predicted.shape))
        predicted = predicted.tolist()
        
        
        # get the attention masks, we need to discard the predictions made for extra padding
        attnMasksBatch = batchData[2]
        predictedTags = []
        logitSm = []
        if attnMasksBatch is not None:
            #shape of attention Masks (batchSize, maxSeqLen)
            actualLengths = attnMasksBatch.cpu().numpy().sum(axis = 1).tolist()
            for i, pred in enumerate(predicted):
                predictedTags.append( pred[:actualLengths[i]] )
                logitSm.append(outLogitsSoftmax[i][:actualLengths[i]])
            return predictedTags, logitSm
        else:
            return predicted, outLogitsSoftmax
       

    def save_multi_task_model(self, savePath):
        '''
        We will save the model parameters with state dict.
        Also the current optimizer state.
        Along with the current global_steps and epoch which would help for resuming training
        Plus we will save the task parameters object created from the task file.
        The same object shall be used for this saved model
        '''
        modelStateDict = {k : v.cpu() for k,v in self.network.state_dict().items()}
        toSave = {'model_state_dict' :modelStateDict,
                'optimizer_state' : self.optimizer.state_dict(),
                'scheduler_state' : self.scheduler.state_dict(),
                'global_step' : self.globalStep,
                'task_params' : self.taskParams}
        
        torch.save(toSave, savePath)
        logger.info('model saved in {} global step at {}'.format(self.globalStep, savePath))

    def load_multi_task_model(self, loadedDict):
        
        '''
        Need to check state dict for multi-gpu and single-gpu compatibility
        '''
        
        loadedDict['model_state_dict'] = {k.lstrip('module.'):v for k, v in loadedDict['model_state_dict'].items()}
        if torch.cuda.device_count() > 1:
            loadedDict['model_state_dict'] = {'module.'+k : v for k, v in loadedDict['model_state_dict'].items()}


        self.network.load_state_dict(loadedDict['model_state_dict'])
        self.optimizer.load_state_dict(loadedDict['optimizer_state'])
        self.scheduler.load_state_dict(loadedDict['scheduler_state'])
        self.globalStep = loadedDict['global_step']

    def load_shared_model(self, loadedDict, freeze):

        loadedDict['model_state_dict'] = {k.lstrip('module.'):v for k, v in loadedDict['model_state_dict'].items()}
        if torch.cuda.device_count() > 1:
            #current network requires 'module'
            loadedDict['model_state_dict'] = {'module.'+k : v for k, v in loadedDict['model_state_dict'].items()}

        # filling in weights from saved shared model to model
        pretrainedDict = loadedDict['model_state_dict']
        pretrainedTaskParams = loadedDict['task_params']
       
        modelDict = self.network.state_dict()
        logger.info('transferring weight of shared model')
        
        updateDict = {k :pretrainedDict[k] for k in modelDict if k.startswith('sharedModel') or k.startswith('poolerLayer')}
        logger.info('number of parameters transferred for shared model {}'.format(len(updateDict)))

        logger.info('looking for common parameters for task specific headers...')
        
        
        for taskId, taskName in pretrainedTaskParams.taskIdNameMap.items():
            for key in modelDict:
                if key.startswith('allHeaders.{}'.format(taskName)):
                    updateDict[key] = pretrainedDict[key]
                    logger.info('transferring parameter for task header: {}'.format(taskName))
        
       
        modelDict.update(updateDict)
        self.network.load_state_dict(modelDict)

        if freeze is True:
            for name, param in self.network.named_parameters():
                if name.startswith('sharedModel'):
                    param.requires_grad = False
            logger.info("shared model weights frozen for finetune..")
