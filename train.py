'''
Final Training script to run traininig for multi-task
'''
import argparse
import random
import numpy as np
import pandas as pd
import logging
import torch
import os
import math
from datetime import datetime
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.task_utils import TasksParam   
from utils.data_utils import METRICS, TaskType

from SRL.data_manager import allTasksDataset, Batcher, batchUtils
from torch.utils.data import Dataset, DataLoader, BatchSampler
from logger_ import make_logger
from SRL.model import multiTaskModel
from SRL.eval import evaluate
TF_ENABLE_ONEDNN_OPTS=0

def make_arguments(parser):
    parser.add_argument('--data_dir', type = str, required=True,
                        help='path to directory where prepared data is present')
    parser.add_argument('--task_file', type = str, required = True,
                        help = 'path to the yml task file')
    parser.add_argument('--out_dir', type = str, required=True,
                        help = 'path to save the model')
    parser.add_argument('--epochs', type = int, required=True,
                        help = 'number of epochs to train')
    parser.add_argument('--freeze_shared_model', default=False, action='store_true',
                        help = "True to freeze the loaded pre-trained shared model and only finetune task specific headers")
    parser.add_argument('--train_batch_size', type = int, default=32,
                        help='batch size to use for training')
    parser.add_argument('--eval_batch_size', type = int, default = 32,
                        help = "batch size to use during evaluation")
    parser.add_argument('--grad_accumulation_steps', type =int, default = 1,
                        help = "number of steps to accumulate gradients before update")
    parser.add_argument('--num_of_warmup_steps', type=int, default = 0,
                        help = "warm-up value for scheduler")
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help = "learning rate for optimizer")
    parser.add_argument('--epsilon', type=float, default=1e-8,
                       help="epsilon value for optimizer")
    parser.add_argument('--grad_clip_value', type = float, default=1.0,
                        help = "gradient clipping value to avoid gradient overflowing" )
    parser.add_argument('--log_file', default='multi_task_logs.log', type = str,
                        help = "name of log file to store")
    parser.add_argument('--log_per_updates', default = 10, type = int,
                        help = "number of steps after which to log loss")
    parser.add_argument('--seed', default=42, type = int,
                        help = "seed to set for modules")
    parser.add_argument('--max_seq_len', default=128, type =int,
                        help = "max seq length used for model at time of data preparation")
    parser.add_argument('--save_per_updates', default = 0, type = int,
                        help = "to keep saving model after this number of updates")
    parser.add_argument('--limit_save', default = 10, type = int,
                        help = "max number recent checkpoints to keep saved")
    parser.add_argument('--load_saved_model', type=str, default=None,
                        help="path to the saved model in case of loading from saved")
    parser.add_argument('--eval_while_train', default = False, action = 'store_true',
                        help = "if evaluation on dev set is required during training.")
    parser.add_argument('--test_while_train', default=False, action = 'store_true',
                        help = "if evaluation on test set is required during training.")
    parser.add_argument('--resume_train', default=False, action = 'store_true',
                        help="Set for resuming training from a saved model")
    parser.add_argument('--finetune', default= False, action = 'store_true',
                        help = "If only the shared model is to be loaded with saved pre-trained multi-task model.\
                            In this case, you can specify your own tasks with task file and use the pre-trained shared model\
                            to finetune upon.")
    parser.add_argument('--debug_mode', default = False, action = 'store_true', 
                        help = "record logs for debugging if True")
    parser.add_argument('--silent', default = False, action = 'store_true', 
                        help = "Only write logs to file if True")
    return parser
    
parser = argparse.ArgumentParser()
parser = make_arguments(parser)
args = parser.parse_args()

#setting logging
now = datetime.now()
logDir = now.strftime("%d_%m-%H_%M")
if not os.path.isdir(logDir):
    os.makedirs(logDir)

logger = make_logger(name = "multi_task", debugMode=args.debug_mode,
                    logFile=os.path.join(logDir, args.log_file), silent=args.silent)
logger.info("logger created.")

device = torch.device('cpu')     
  
#setting seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')

assert os.path.isdir(args.data_dir), "data_dir doesn't exists"
assert os.path.exists(args.task_file), "task_file doesn't exists"
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)


def make_data_handlers(taskParams, mode, isTrain, gpu):
    '''
    This function makes the allTaskDataset, Batch Sampler, Collater function
    and DataLoader for train, dev and test files as per mode.
    In order of task file, 
    train file is at 0th index
    dev file is at 1st index
    test file at 2nd index
    '''
    modePosMap = {"train" : 0, "dev" : 1, "test" : 2, 
                            "abolish_dev" : 3, "abolish_test" : 4, 
                            "alter_dev" : 5, "alter_test" : 6,
                            "begin_dev" : 7, "begin_test" : 8,
                            "block_dev" : 9, "block_test" : 10,
                            "catalyse_dev" : 11, "catalyse_test" : 12,
                            "confer_dev" : 13, "confer_test" : 14,
                            "decrease_dev" : 15, "decrease_test" : 16,
                            "delete_dev" : 17, "delete_test" : 18,
                            "develop_dev" : 19, "develop_test" : 20,
                            "disrupt_dev" : 21, "disrupt_test" : 22,
                            "eliminate_dev" : 23, "eliminate_test" : 24,
                            "encode_dev" : 25, "encode_test" : 26,
                            "express_dev" : 27, "express_test" : 28,
                            "generate_dev" : 29, "generate_test" : 30,
                            "inhibit_dev" : 31, "inhibit_test" : 32,
                            "initiate_dev" : 33, "initiate_test" : 34,
                            "lead_dev" : 35, "lead_test" : 36,
                            "lose_dev" : 37, "lose_test" : 38,
                            "modify_dev" : 39, "modify_test" : 40,
                            "mutate_dev" : 41, "mutate_test" : 42,
                            "proliferate_dev" : 43, "proliferate_test" : 44,
                            "recognize_dev" : 45, "recognize_test" : 46,
                            "result_dev" : 47, "result_test" : 48,
                            "skip_dev" : 49, "skip_test" : 50,
                            "splice_dev" : 51, "splice_test" : 52,
                            "transcribe_dev" : 53, "transcribe_test" : 54,
                            "transform_dev" : 55, "transform_test" : 56,
                            "translate_dev" : 57, "translate_test" : 58,
                            "truncate_dev" : 59, "truncate_test" : 60  
                            }
 
    modeIdx = modePosMap[mode]
    allTaskslist = []
    for taskId, taskName in taskParams.taskIdNameMap.items():
        taskType = taskParams.taskTypeMap[taskName]
        if mode == "test":
            assert len(taskParams.fileNamesMap[taskName])!=100, "test file is required along with train, dev"
        #dataFileName =  '{}.json'.format(taskParams.fileNamesMap[taskName][modeIdx].split('.')[0])
        dataFileName = '{}.json'.format(taskParams.fileNamesMap[taskName][modeIdx].lower().replace('.tsv',''))
        taskDataPath = os.path.join(args.data_dir, dataFileName)
        assert os.path.exists(taskDataPath), "{} doesn't exist".format(taskDataPath)
        taskDict = {"data_task_id" : int(taskId),
                    "data_path" : taskDataPath,
                    "data_task_type" : taskType,
                    "data_task_name" : taskName}
        allTaskslist.append(taskDict)

    allData = allTasksDataset(allTaskslist)
   
    if mode == "train":
        batchSize = args.train_batch_size
    else:
        batchSize = args.eval_batch_size

    batchSampler = Batcher(allData, batchSize=batchSize, seed = args.seed)
    batchSamplerUtils = batchUtils(isTrain = isTrain, modelType= taskParams.modelType,
                                  maxSeqLen = args.max_seq_len)
    multiTaskDataLoader = DataLoader(allData, batch_sampler = batchSampler,
                                collate_fn=batchSamplerUtils.collate_fn,
                                pin_memory=gpu)

    return allData, batchSampler, multiTaskDataLoader

def main():
    allParams = vars(args)
    logger.info('ARGS : {}'.format(allParams))
    # loading if load_saved_model
    if args.load_saved_model is not None:
        assert os.path.exists(args.load_saved_model), "saved model not present at {}".format(args.load_saved_model)
        loadedDict = torch.load(args.load_saved_model, map_location=device)
        logger.info('Saved Model loaded from {}'.format(args.load_saved_model))

        if args.finetune is True:
            '''
            NOTE :- 
            In finetune mode, only the weights from the shared encoder (pre-trained) from the model will be used. The headers
            over the model will be made from the task file. You can further finetune for training the entire model.
            Freezing of the pre-trained moddel is also possible with argument 
            '''
            logger.info('In Finetune model. Only shared Encoder weights will be loaded from {}'.format(args.load_saved_model))
            logger.info('Task specific headers will be made according to task file')
            taskParams = TasksParam(args.task_file)

        else:
            '''
            NOTE : -
            taskParams used with this saved model must also be stored. THE SAVED TASK PARAMS 
            SHALL BE USED HERE TO AVOID ANY DISCREPENCIES/CHANGES IN THE TASK FILE.
            Hence, if changes are made to task file after saving this model, they shall be ignored
            '''
            taskParams = loadedDict['task_params']
            logger.info('Task Params loaded from saved model.')
            logger.info('Any changes made to task file except the data \
                        file paths after saving this model shall be ignored')
            tempTaskParams = TasksParam(args.task_file)
            #transfering the names of file in new task file to loaded task params
            for taskId, taskName in taskParams.taskIdNameMap.items():
                assert taskName in tempTaskParams.taskIdNameMap.values(), "task names changed in task file given.\
                tasks supported for loaded model are {}".format(list(taskParams.taskIdNameMap.values()))

                taskParams.fileNamesMap[taskName] = tempTaskParams.fileNamesMap[taskName]
    else:
        taskParams = TasksParam(args.task_file)
        logger.info("Task params object created from task file...")
        

    allParams['task_params'] = taskParams
    allParams['gpu'] = torch.cuda.is_available()
    logger.info('task parameters:\n {}'.format(taskParams.taskDetails))

    tensorboard = SummaryWriter(log_dir = os.path.join(logDir, 'tb_logs'))
    logger.info("Tensorboard writing at {}".format(os.path.join(logDir, 'tb_logs')))

    # making handlers for train
    logger.info("Creating data handlers for training...")
    allDataTrain, BatchSamplerTrain, multiTaskDataLoaderTrain = make_data_handlers(taskParams,
                                                                                "train", isTrain=True,
                                                                                gpu = allParams['gpu'])
    # if evaluation on dev set is required during training. Labels are required
    # It will occur at the end of each epoch
    if args.eval_while_train:
        logger.info("Creating data handlers for dev...")
        allDataDev, BatchSamplerDev, multiTaskDataLoaderDev = make_data_handlers(taskParams,
                                                                                "dev", isTrain=False,
                                                                                gpu=allParams['gpu'])
    # if evaluation on test set is required during training. Labels are required
    # It will occur at the end of each epoch
    if args.test_while_train:  
        logger.info("Creating data handlers for test...")
        allDataTest, BatchSamplerTest, multiTaskDataLoaderTest = make_data_handlers(taskParams,
                                                                                "test", isTrain=False,
                                                                                gpu=allParams['gpu'])
    
    #Abolish 3 4
    logger.info("Creating data handlers for Abolish dev...")
    allDataAbolish_dev, BatchSamplerAbolish_dev, multiTaskDataLoaderAbolish_dev = make_data_handlers(taskParams,
                                                                                "abolish_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Abolish test...")
    allDataAbolish_test, BatchSamplerAbolish_test, multiTaskDataLoaderAbolish_test = make_data_handlers(taskParams,
                                                                                "abolish_test", isTrain=False,
                                                                                gpu=allParams['gpu'])

    #Alter 5 6
    logger.info("Creating data handlers for Alter dev...")
    allDataAlter_dev, BatchSamplerAlter_dev, multiTaskDataLoaderAlter_dev = make_data_handlers(taskParams,
                                                                                "alter_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Alter test...")
    allDataAlter_test, BatchSamplerAlter_test, multiTaskDataLoaderAlter_test = make_data_handlers(taskParams,
                                                                                "alter_test", isTrain=False,
                                                                                gpu=allParams['gpu'])
    #Begin 7 8
    logger.info("Creating data handlers for Begin dev...")
    allDataBegin_dev, BatchSamplerBegin_dev, multiTaskDataLoaderBegin_dev = make_data_handlers(taskParams,
                                                                                "begin_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Begin test...")
    allDataBegin_test, BatchSamplerBegin_test, multiTaskDataLoaderBegin_test = make_data_handlers(taskParams,
                                                                                "begin_test", isTrain=False,
                                                                                gpu=allParams['gpu'])
    
    #Block 9 10
    logger.info("Creating data handlers for Block dev...")
    allDataBlock_dev, BatchSamplerBlock_dev, multiTaskDataLoaderBlock_dev = make_data_handlers(taskParams,
                                                                                "block_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Block test...")
    allDataBlock_test, BatchSamplerBlock_test, multiTaskDataLoaderBlock_test = make_data_handlers(taskParams,
                                                                                "block_test", isTrain=False,
                                                                                gpu=allParams['gpu'])

    #Catalyse 11 12
    logger.info("Creating data handlers for Catalyse dev...")
    allDataCatalyse_dev, BatchSamplerCatalyse_dev, multiTaskDataLoaderCatalyse_dev = make_data_handlers(taskParams,
                                                                                "catalyse_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Catalyse test...")
    allDataCatalyse_test, BatchSamplerCatalyse_test, multiTaskDataLoaderCatalyse_test = make_data_handlers(taskParams,
                                                                                "catalyse_test", isTrain=False,
                                                                                gpu=allParams['gpu'])
    #Confer 13 14
    logger.info("Creating data handlers for Confer dev...")
    allDataConfer_dev, BatchSamplerConfer_dev, multiTaskDataLoaderConfer_dev = make_data_handlers(taskParams,
                                                                                "confer_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Confer test...")
    allDataConfer_test, BatchSamplerConfer_test, multiTaskDataLoaderConfer_test = make_data_handlers(taskParams,
                                                                                "confer_test", isTrain=False,
                                                                                gpu=allParams['gpu'])

    #Decrease 15 16
    logger.info("Creating data handlers for Decrease dev...")
    allDataDecrease_dev, BatchSamplerDecrease_dev, multiTaskDataLoaderDecrease_dev = make_data_handlers(taskParams,
                                                                                "decrease_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Decrease test...")
    allDataDecrease_test, BatchSamplerDecrease_test, multiTaskDataLoaderDecrease_test = make_data_handlers(taskParams,
                                                                                "decrease_test", isTrain=False,
                                                                                gpu=allParams['gpu'])

    #Delete 17 18
    logger.info("Creating data handlers for Delete dev...")
    allDataDelete_dev, BatchSamplerDelete_dev, multiTaskDataLoaderDelete_dev = make_data_handlers(taskParams,
                                                                                "delete_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Delete test...")
    allDataDelete_test, BatchSamplerDelete_test, multiTaskDataLoaderDelete_test = make_data_handlers(taskParams,
                                                                                "delete_test", isTrain=False,
                                                                                gpu=allParams['gpu'])

    #Develop 19 20
    logger.info("Creating data handlers for Develop dev...")
    allDataDevelop_dev, BatchSamplerDevelop_dev, multiTaskDataLoaderDevelop_dev = make_data_handlers(taskParams,
                                                                                "develop_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Develop test...")
    allDataDevelop_test, BatchSamplerDevelop_test, multiTaskDataLoaderDevelop_test = make_data_handlers(taskParams,
                                                                                "develop_test", isTrain=False,
                                                                                gpu=allParams['gpu'])

    #Disrupt 21 22
    logger.info("Creating data handlers for Disrupt dev...")
    allDataDisrupt_dev, BatchSamplerDisrupt_dev, multiTaskDataLoaderDisrupt_dev = make_data_handlers(taskParams,
                                                                                "disrupt_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Disrupt test...")
    allDataDisrupt_test, BatchSamplerDisrupt_test, multiTaskDataLoaderDisrupt_test = make_data_handlers(taskParams,
                                                                                "disrupt_test", isTrain=False,
                                                                                gpu=allParams['gpu'])
                                                                       
    #Eliminate 23 24
    logger.info("Creating data handlers for Eliminate dev...")
    allDataEliminate_dev, BatchSamplerEliminate_dev, multiTaskDataLoaderEliminate_dev = make_data_handlers(taskParams,
                                                                                "eliminate_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Eliminate test...")
    allDataEliminate_test, BatchSamplerEliminate_test, multiTaskDataLoaderEliminate_test = make_data_handlers(taskParams,
                                                                                "eliminate_test", isTrain=False,
                                                                                gpu=allParams['gpu'])

    #Encode 25 26
    logger.info("Creating data handlers for Encode dev...")
    allDataEncode_dev, BatchSamplerEncode_dev, multiTaskDataLoaderEncode_dev = make_data_handlers(taskParams,
                                                                                "encode_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Encode test...")
    allDataEncode_test, BatchSamplerEncode_test, multiTaskDataLoaderEncode_test = make_data_handlers(taskParams,
                                                                                "encode_test", isTrain=False,
                                                                                gpu=allParams['gpu'])

     #Express 27 28
    logger.info("Creating data handlers for Express dev...")
    allDataExpress_dev, BatchSamplerExpress_dev, multiTaskDataLoaderExpress_dev = make_data_handlers(taskParams,
                                                                                "express_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Express test...")
    allDataExpress_test, BatchSamplerExpress_test, multiTaskDataLoaderExpress_test = make_data_handlers(taskParams,
                                                                                "express_test", isTrain=False,
                                                                                gpu=allParams['gpu'])

    #Generate 29 30
    logger.info("Creating data handlers for Generate dev...")
    allDataGenerate_dev, BatchSamplerGenerate_dev, multiTaskDataLoaderGenerate_dev = make_data_handlers(taskParams,
                                                                                "generate_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Generate test...")
    allDataGenerate_test, BatchSamplerGenerate_test, multiTaskDataLoaderGenerate_test = make_data_handlers(taskParams,
                                                                                "generate_test", isTrain=False,
                                                                                gpu=allParams['gpu'])      

    #Inhibit 31 32
    logger.info("Creating data handlers for Inhibit dev...")
    allDataInhibit_dev, BatchSamplerInhibit_dev, multiTaskDataLoaderInhibit_dev = make_data_handlers(taskParams,
                                                                                "inhibit_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Inhibit test...")
    allDataInhibit_test, BatchSamplerInhibit_test, multiTaskDataLoaderInhibit_test = make_data_handlers(taskParams,
                                                                                "inhibit_test", isTrain=False,
                                                                                gpu=allParams['gpu'])         

    #Initiate 33 34
    logger.info("Creating data handlers for Initiate dev...")
    allDataInitiate_dev, BatchSamplerInitiate_dev, multiTaskDataLoaderInitiate_dev = make_data_handlers(taskParams,
                                                                                "initiate_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Initiate test...")
    allDataInitiate_test, BatchSamplerInitiate_test, multiTaskDataLoaderInitiate_test = make_data_handlers(taskParams,
                                                                                "initiate_test", isTrain=False,
                                                                                gpu=allParams['gpu'])                                                                                                                                                                                                                      

    #Lead 35 36
    logger.info("Creating data handlers for Lead dev...")
    allDataLead_dev, BatchSamplerLead_dev, multiTaskDataLoaderLead_dev = make_data_handlers(taskParams,
                                                                                "lead_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Lead test...")
    allDataLead_test, BatchSamplerLead_test, multiTaskDataLoaderLead_test = make_data_handlers(taskParams,
                                                                                "lead_test", isTrain=False,
                                                                                gpu=allParams['gpu'])      

    #Lose 37 38
    logger.info("Creating data handlers for Lose dev...")
    allDataLose_dev, BatchSamplerLose_dev, multiTaskDataLoaderLose_dev = make_data_handlers(taskParams,
                                                                                "lose_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Lose test...")
    allDataLose_test, BatchSamplerLose_test, multiTaskDataLoaderLose_test = make_data_handlers(taskParams,
                                                                                "lose_test", isTrain=False,
                                                                                gpu=allParams['gpu'])  

    #Modify 39 40
    logger.info("Creating data handlers for Modify dev...")
    allDataModify_dev, BatchSamplerModify_dev, multiTaskDataLoaderModify_dev = make_data_handlers(taskParams,
                                                                                "modify_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Modify test...")
    allDataModify_test, BatchSamplerModify_test, multiTaskDataLoaderModify_test = make_data_handlers(taskParams,
                                                                                "modify_test", isTrain=False,
                                                                                gpu=allParams['gpu'])                                                                                  

    #Mutate 41 42
    logger.info("Creating data handlers for Mutate dev...")
    allDataMutate_dev, BatchSamplerMutate_dev, multiTaskDataLoaderMutate_dev = make_data_handlers(taskParams,
                                                                                "mutate_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Mutate test...")
    allDataMutate_test, BatchSamplerMutate_test, multiTaskDataLoaderMutate_test = make_data_handlers(taskParams,
                                                                                "mutate_test", isTrain=False,
                                                                                gpu=allParams['gpu'])  

    #Proliferate 43 44
    logger.info("Creating data handlers for Proliferate dev...")
    allDataProliferate_dev, BatchSamplerProliferate_dev, multiTaskDataLoaderProliferate_dev = make_data_handlers(taskParams,
                                                                                "proliferate_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Proliferate test...")
    allDataProliferate_test, BatchSamplerProliferate_test, multiTaskDataLoaderProliferate_test = make_data_handlers(taskParams,
                                                                                "proliferate_test", isTrain=False,
                                                                                gpu=allParams['gpu'])         

    #Recognize 45 46
    logger.info("Creating data handlers for Recognize dev...")
    allDataRecognize_dev, BatchSamplerRecognize_dev, multiTaskDataLoaderRecognize_dev = make_data_handlers(taskParams,
                                                                                "recognize_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Proliferate test...")
    allDataRecognize_test, BatchSamplerRecognize_test, multiTaskDataLoaderRecognize_test = make_data_handlers(taskParams,
                                                                                "recognize_test", isTrain=False,
                                                                                gpu=allParams['gpu'])  

    #Result 47 48
    logger.info("Creating data handlers for Result dev...")
    allDataResult_dev, BatchSamplerResult_dev, multiTaskDataLoaderResult_dev = make_data_handlers(taskParams,
                                                                                "result_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Result test...")
    allDataResult_test, BatchSamplerResult_test, multiTaskDataLoaderResult_test = make_data_handlers(taskParams,
                                                                                "result_test", isTrain=False,
                                                                                gpu=allParams['gpu']) 

    #Skip 49 50
    logger.info("Creating data handlers for Skip dev...")
    allDataSkip_dev, BatchSamplerSkip_dev, multiTaskDataLoaderSkip_dev = make_data_handlers(taskParams,
                                                                                "skip_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Skip test...")
    allDataSkip_test, BatchSamplerSkip_test, multiTaskDataLoaderSkip_test = make_data_handlers(taskParams,
                                                                                "skip_test", isTrain=False,
                                                                                gpu=allParams['gpu']) 

     #Splice 51 52
    logger.info("Creating data handlers for Splice dev...")
    allDataSplice_dev, BatchSamplerSplice_dev, multiTaskDataLoaderSplice_dev = make_data_handlers(taskParams,
                                                                                "splice_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Splice test...")
    allDataSplice_test, BatchSamplerSplice_test, multiTaskDataLoaderSplice_test = make_data_handlers(taskParams,
                                                                                "splice_test", isTrain=False,
                                                                                gpu=allParams['gpu']) 


    #Transcribe 53 54
    logger.info("Creating data handlers for Transcribe dev...")
    allDataTranscribe_dev, BatchSamplerTranscribe_dev, multiTaskDataLoaderTranscribe_dev = make_data_handlers(taskParams,
                                                                                "transcribe_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Transcribe test...")
    allDataTranscribe_test, BatchSamplerTranscribe_test, multiTaskDataLoaderTranscribe_test = make_data_handlers(taskParams,
                                                                                "transcribe_test", isTrain=False,
                                                                                gpu=allParams['gpu'])                                                                             

    #Transform 55 56
    logger.info("Creating data handlers for Transform dev...")
    allDataTransform_dev, BatchSamplerTransform_dev, multiTaskDataLoaderTransform_dev = make_data_handlers(taskParams,
                                                                                "transform_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Transform test...")
    allDataTransform_test, BatchSamplerTransform_test, multiTaskDataLoaderTransform_test = make_data_handlers(taskParams,
                                                                                "transform_test", isTrain=False,
                                                                                gpu=allParams['gpu']) 


    #Translate 57 58
    logger.info("Creating data handlers for Translate dev...")
    allDataTranslate_dev, BatchSamplerTranslate_dev, multiTaskDataLoaderTranslate_dev = make_data_handlers(taskParams,
                                                                                "translate_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Translate test...")
    allDataTranslate_test, BatchSamplerTranslate_test, multiTaskDataLoaderTranslate_test = make_data_handlers(taskParams,
                                                                                "translate_test", isTrain=False,
                                                                                gpu=allParams['gpu'])                       

    #Truncate 59 60
    logger.info("Creating data handlers for Truncate dev...")
    allDataTruncate_dev, BatchSamplerTruncate_dev, multiTaskDataLoaderTruncate_dev = make_data_handlers(taskParams,
                                                                                "transform_dev", isTrain=False,
                                                                               gpu=allParams['gpu'])
    logger.info("Creating data handlers for Truncate test...")
    allDataTruncate_test, BatchSamplerTruncate_test, multiTaskDataLoaderTruncate_test = make_data_handlers(taskParams,
                                                                                "truncate_test", isTrain=False,
                                                                                gpu=allParams['gpu']) 

    #making multi-task model 
    allParams['num_train_steps'] = math.ceil(len(multiTaskDataLoaderTrain)/args.train_batch_size) *args.epochs // args.grad_accumulation_steps
    
    allParams['warmup_steps'] = args.num_of_warmup_steps
    allParams['learning_rate'] = args.learning_rate
    allParams['epsilon'] = args.epsilon
    
    logger.info("NUM TRAIN STEPS: {}".format(allParams['num_train_steps']))
    logger.info("len of dataloader: {}".format(len(multiTaskDataLoaderTrain)))
    logger.info("Making multi-task model...")
    model = multiTaskModel(allParams)
    print(model)
    
    if args.load_saved_model:
        if args.finetune is True:
            model.load_shared_model(loadedDict, args.freeze_shared_model)
            
            logger.info('shared model loaded for finetune from {}'.format(args.load_saved_model))
        else:
            model.load_multi_task_model(loadedDict)
            logger.info('saved model loaded with global step {} from {}'.format(model.globalStep,
                                                                            args.load_saved_model))
        if args.resume_train:
            logger.info("Resuming training from global step {}. Steps before it will be skipped".format(model.globalStep))
        
    globalStep = 0    
    check_point = model.globalStep
    # training 
    resCnt = 0
    for epoch in range(args.epochs):
        logger.info('\n####################### EPOCH {} ###################\n'.format(epoch))
        totalEpochLoss = 0
        text = "Epoch: {}".format(epoch)
        tt = int(allParams['num_train_steps']*args.grad_accumulation_steps/args.epochs)
       
        with tqdm(total = tt, position=epoch, desc=text) as progress:
            
            for i, (batchMetaData, batchData) in enumerate(multiTaskDataLoaderTrain):
                batchMetaData, batchData = BatchSamplerTrain.patch_data(batchMetaData,batchData, gpu = allParams['gpu'])
                
                if args.resume_train and args.load_saved_model and resCnt*args.grad_accumulation_steps <= check_point:
                    '''
                    NOTE: - Resume function is only to be used in case the training process couldnt
                    complete or you wish to extend the training to some more epochs.
                    Please keep the gradient accumulation step the same for exact resuming.
                    '''
                    resCnt += 1
                    progress.update(1)
                    continue
                model.update_step(batchMetaData, batchData)
                
                totalEpochLoss += model.taskLoss.item()

                if model.globalStep % args.log_per_updates == 0 and (model.accumulatedStep+1 == args.grad_accumulation_steps):
                    taskId = batchMetaData['task_id']
                    taskName = taskParams.taskIdNameMap[taskId]
                    avgLoss = totalEpochLoss / (i+1)
                   
                    tensorboard.add_scalar('train/avg_loss', avgLoss, global_step= model.globalStep)
                    tensorboard.add_scalar('train/{}_loss'.format(taskName),
                                            model.taskLoss.item(),
                                            global_step=model.globalStep)
                
                if args.save_per_updates > 0 and  ((model.globalStep+1) % args.save_per_updates)==0 and (model.accumulatedStep+1==args.grad_accumulation_steps):
                    savePath = os.path.join(args.out_dir, 'multi_task_model_{}_{}.pt'.format(epoch,
                                                                                            model.globalStep))
                    model.save_multi_task_model(savePath)
                    
                    # limiting the checkpoints save, remove checkpoints if beyond limit
                    if args.limit_save > 0:
                        stepCkpMap = {int(ckp.rstrip('.pt').split('_')[-1]) : ckp for ckp in os.listdir(args.out_dir) if ckp.endswith('.pt') }
                        
                        #sorting based on global step
                        stepToDel = sorted(list(stepCkpMap.keys()))[:-args.limit_save]

                        for ckpStep in stepToDel:
                            os.remove(os.path.join(args.out_dir, stepCkpMap[ckpStep]))
                            logger.info('Removing checkpoint {}'.format(stepCkpMap[ckpStep]))

                progress.update(1)
            globalStep += int(tt/args.grad_accumulation_steps)
          
            #saving model after epoch
            if args.resume_train and args.load_saved_model and resCnt*args.grad_accumulation_steps <= check_point:
                
                pass
            else:
                model.globalStep = globalStep
            
                savePath = os.path.join(args.out_dir, 'multi_task_model_{}_{}.pt'.format(epoch, model.globalStep))  
                model.save_multi_task_model(savePath)
                

            if args.eval_while_train:
                logger.info("\nRunning Evaluation on dev...")
                with torch.no_grad():
                    evaluate(allDataDev, BatchSamplerDev, multiTaskDataLoaderDev, taskParams,
                            model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)

            if args.test_while_train:
                logger.info("\nRunning Evaluation on test...")
                wrtPredpath = "test_predictions_{}.tsv".format(epoch)
                with torch.no_grad():
                    evaluate(allDataTest, BatchSamplerTest, multiTaskDataLoaderTest, taskParams,
                            model, gpu=allParams['gpu'], evalBatchSize = args.eval_batch_size, needMetrics=True, hasTrueLabels=True,
                            wrtDir=args.out_dir, wrtPredPath=wrtPredpath)

            # Abolish 3 4 
            logger.info("\nRunning Evaluation on Abolish dev...")
            with torch.no_grad():
                evaluate(allDataAbolish_dev, BatchSamplerAbolish_dev, multiTaskDataLoaderAbolish_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Abolish test...")
            with torch.no_grad():
                evaluate(allDataAbolish_test, BatchSamplerAbolish_test, multiTaskDataLoaderAbolish_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            # ALter 5 6
            logger.info("\nRunning Evaluation on Alter dev...")
            with torch.no_grad():
                evaluate(allDataAlter_dev, BatchSamplerAlter_dev, multiTaskDataLoaderAlter_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Alter test...")
            with torch.no_grad():
                evaluate(allDataAlter_test, BatchSamplerAlter_test, multiTaskDataLoaderAlter_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)

            #Begin 7 8
            logger.info("\nRunning Evaluation on Begin dev...")
            with torch.no_grad():
                evaluate(allDataBegin_dev, BatchSamplerBegin_dev, multiTaskDataLoaderBegin_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Begin test...")
            with torch.no_grad():
                evaluate(allDataBegin_test, BatchSamplerBegin_test, multiTaskDataLoaderBegin_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
                
        
            #Block 9 10
            logger.info("\nRunning Evaluation on Block dev...")
            with torch.no_grad():
                evaluate(allDataBlock_dev, BatchSamplerBlock_dev, multiTaskDataLoaderBlock_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Block test...")
            with torch.no_grad():
                evaluate(allDataBlock_test, BatchSamplerBlock_test, multiTaskDataLoaderBlock_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)

            #Catalyse 11 12
            logger.info("\nRunning Evaluation on Catalyse dev...")
            with torch.no_grad():
                evaluate(allDataCatalyse_dev, BatchSamplerCatalyse_dev, multiTaskDataLoaderCatalyse_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Catalyse test...")
            with torch.no_grad():
                evaluate(allDataCatalyse_test, BatchSamplerCatalyse_test, multiTaskDataLoaderCatalyse_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)        

            #Confer 13 14
            logger.info("\nRunning Evaluation on Confer dev...")
            with torch.no_grad():
                evaluate(allDataConfer_dev, BatchSamplerConfer_dev, multiTaskDataLoaderConfer_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Confer test...")
            with torch.no_grad():
                evaluate(allDataConfer_test, BatchSamplerConfer_test, multiTaskDataLoaderConfer_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            #Decrease 15 16
            logger.info("\nRunning Evaluation on Decrease dev...")
            with torch.no_grad():
                evaluate(allDataDecrease_dev, BatchSamplerDecrease_dev, multiTaskDataLoaderDecrease_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Decrease test...")
            with torch.no_grad():
                evaluate(allDataDecrease_test, BatchSamplerDecrease_test, multiTaskDataLoaderDecrease_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
                
        
            #Delete 17 18
            logger.info("\nRunning Evaluation on Delete dev...")
            with torch.no_grad():
                evaluate(allDataDelete_dev, BatchSamplerDelete_dev, multiTaskDataLoaderDelete_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Delete test...")
            with torch.no_grad():
                evaluate(allDataDelete_test, BatchSamplerDelete_test, multiTaskDataLoaderDelete_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            #Develop 19 20
            logger.info("\nRunning Evaluation on Develop dev...")
            with torch.no_grad():
                evaluate(allDataDevelop_dev, BatchSamplerDevelop_dev, multiTaskDataLoaderDevelop_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Develop test...")
            with torch.no_grad():
                evaluate(allDataDevelop_test, BatchSamplerDevelop_test, multiTaskDataLoaderDevelop_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)

            #Disrupt 21 22
            logger.info("\nRunning Evaluation on Disrupt dev...")
            with torch.no_grad():
                evaluate(allDataDisrupt_dev, BatchSamplerDisrupt_dev, multiTaskDataLoaderDisrupt_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Disrupt test...")
            with torch.no_grad():
                evaluate(allDataDisrupt_test, BatchSamplerDisrupt_test, multiTaskDataLoaderDisrupt_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)

            #Eliminate 23 24
            logger.info("\nRunning Evaluation on Eliminate dev...")
            with torch.no_grad():
                evaluate(allDataEliminate_dev, BatchSamplerEliminate_dev, multiTaskDataLoaderEliminate_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Eliminate test...")
            with torch.no_grad():
                evaluate(allDataEliminate_test, BatchSamplerEliminate_test, multiTaskDataLoaderEliminate_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            #Encode 25 26
            logger.info("\nRunning Evaluation on Encode dev...")
            with torch.no_grad():
                evaluate(allDataEncode_dev, BatchSamplerEncode_dev, multiTaskDataLoaderEncode_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Encode test...")
            with torch.no_grad():
                evaluate(allDataEncode_test, BatchSamplerEncode_test, multiTaskDataLoaderEncode_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)

            #Express 27 28
            logger.info("\nRunning Evaluation on Express dev...")
            with torch.no_grad():
                evaluate(allDataExpress_dev, BatchSamplerExpress_dev, multiTaskDataLoaderExpress_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Express test...")
            with torch.no_grad():
                evaluate(allDataExpress_test, BatchSamplerExpress_test, multiTaskDataLoaderExpress_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            #Generate 29 30
            logger.info("\nRunning Evaluation on Generate dev...")
            with torch.no_grad():
                evaluate(allDataGenerate_dev, BatchSamplerGenerate_dev, multiTaskDataLoaderGenerate_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Generate test...")
            with torch.no_grad():
                evaluate(allDataGenerate_test, BatchSamplerGenerate_test, multiTaskDataLoaderGenerate_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            #Inhibit 31 32
            logger.info("\nRunning Evaluation on Inhibit dev...")
            with torch.no_grad():
                evaluate(allDataInhibit_dev, BatchSamplerInhibit_dev, multiTaskDataLoaderInhibit_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Inhibit test...")
            with torch.no_grad():
                evaluate(allDataInhibit_test, BatchSamplerInhibit_test, multiTaskDataLoaderInhibit_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            #Initiate 33 34
            logger.info("\nRunning Evaluation on Initiate dev...")
            with torch.no_grad():
                evaluate(allDataInitiate_dev, BatchSamplerInitiate_dev, multiTaskDataLoaderInitiate_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Initiate test...")
            with torch.no_grad():
                evaluate(allDataInitiate_test, BatchSamplerInitiate_test, multiTaskDataLoaderInitiate_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            #Lead 35 36
            logger.info("\nRunning Evaluation on Lead dev...")
            with torch.no_grad():
                evaluate(allDataLead_dev, BatchSamplerLead_dev, multiTaskDataLoaderLead_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Lead test...")
            with torch.no_grad():
                evaluate(allDataLead_test, BatchSamplerLead_test, multiTaskDataLoaderLead_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)

            #Lose 37 38
            logger.info("\nRunning Evaluation on Lose dev...")
            with torch.no_grad():
                evaluate(allDataLose_dev, BatchSamplerLose_dev, multiTaskDataLoaderLose_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Lose test...")
            with torch.no_grad():
                evaluate(allDataLose_test, BatchSamplerLose_test, multiTaskDataLoaderLose_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)

            #Modify 39 40
            logger.info("\nRunning Evaluation on Modify dev...")
            with torch.no_grad():
                evaluate(allDataModify_dev, BatchSamplerModify_dev, multiTaskDataLoaderModify_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Modify test...")
            with torch.no_grad():
                evaluate(allDataModify_test, BatchSamplerModify_test, multiTaskDataLoaderModify_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)

            #Mutate 41 42
            logger.info("\nRunning Evaluation on Mutate dev...")
            with torch.no_grad():
                evaluate(allDataMutate_dev, BatchSamplerMutate_dev, multiTaskDataLoaderMutate_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Mutate test...")
            with torch.no_grad():
                evaluate(allDataMutate_test, BatchSamplerMutate_test, multiTaskDataLoaderMutate_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)

            #Proliferate 43 44
            logger.info("\nRunning Evaluation on Proliferate dev...")
            with torch.no_grad():
                evaluate(allDataProliferate_dev, BatchSamplerProliferate_dev, multiTaskDataLoaderProliferate_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Proliferate test...")
            with torch.no_grad():
                evaluate(allDataProliferate_test, BatchSamplerProliferate_test, multiTaskDataLoaderProliferate_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)

            #Recognize 45 46
            logger.info("\nRunning Evaluation on Recognize dev...")
            with torch.no_grad():
                evaluate(allDataRecognize_dev, BatchSamplerRecognize_dev, multiTaskDataLoaderRecognize_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Recognize test...")
            with torch.no_grad():
                evaluate(allDataRecognize_test, BatchSamplerRecognize_test, multiTaskDataLoaderRecognize_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)

            #Result 47 48
            logger.info("\nRunning Evaluation on Result dev...")
            with torch.no_grad():
                evaluate(allDataResult_dev, BatchSamplerResult_dev, multiTaskDataLoaderResult_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Result test...")
            with torch.no_grad():
                evaluate(allDataResult_test, BatchSamplerResult_test, multiTaskDataLoaderResult_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            #Skip 49 50
            logger.info("\nRunning Evaluation on Skip dev...")
            with torch.no_grad():
                evaluate(allDataSkip_dev, BatchSamplerSkip_dev, multiTaskDataLoaderSkip_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Skip test...")
            with torch.no_grad():
                evaluate(allDataSkip_test, BatchSamplerSkip_test, multiTaskDataLoaderSkip_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            #Splice 51 52
            logger.info("\nRunning Evaluation on Splice dev...")
            with torch.no_grad():
                evaluate(allDataSplice_dev, BatchSamplerSplice_dev, multiTaskDataLoaderSplice_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Splice test...")
            with torch.no_grad():
                evaluate(allDataSplice_test, BatchSamplerSplice_test, multiTaskDataLoaderSplice_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
          
            #Transcribe 53 54
            logger.info("\nRunning Evaluation on Transcribe dev...")
            with torch.no_grad():
                evaluate(allDataTranscribe_dev, BatchSamplerTranscribe_dev, multiTaskDataLoaderTranscribe_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Transcribe test...")
            with torch.no_grad():
                evaluate(allDataTranscribe_test, BatchSamplerTranscribe_test, multiTaskDataLoaderTranscribe_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            #Transform 55 56
            logger.info("\nRunning Evaluation on Transform dev...")
            with torch.no_grad():
                evaluate(allDataTransform_dev, BatchSamplerTransform_dev, multiTaskDataLoaderTransform_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Transform test...")
            with torch.no_grad():
                evaluate(allDataTransform_test, BatchSamplerTransform_test, multiTaskDataLoaderTransform_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
                
            #Translate 57 58
            logger.info("\nRunning Evaluation on Translate dev...")
            with torch.no_grad():
                evaluate(allDataTranslate_dev, BatchSamplerTranslate_dev, multiTaskDataLoaderTranslate_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Translate test...")
            with torch.no_grad():
                evaluate(allDataTranslate_test, BatchSamplerTranslate_test, multiTaskDataLoaderTranslate_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
                
        
            #Truncate 59 60
            logger.info("\nRunning Evaluation on Truncate dev...")
            with torch.no_grad():
                evaluate(allDataTruncate_dev, BatchSamplerTruncate_dev, multiTaskDataLoaderTruncate_dev, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
            
            logger.info("\nRunning Evaluation on Truncate test...")
            with torch.no_grad():
                evaluate(allDataTruncate_test, BatchSamplerTruncate_test, multiTaskDataLoaderTruncate_test, taskParams,
                    model, gpu=allParams['gpu'],evalBatchSize=args.eval_batch_size, hasTrueLabels=True, needMetrics=True)
if __name__ == "__main__":
    main()