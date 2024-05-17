import json
from tqdm import tqdm
import argparse
import os
import multiprocessing as mp
from ast import literal_eval
from utils.task_utils import TasksParam
from utils.data_utils import TaskType, NLP_MODELS
from keras_preprocessing.sequence import pad_sequences

def load_data(dataPath, hasLabels):
    '''
    This fn loads data from tsv file in according to the format in taskType
    dataPath - path/name of file to read
    taskType - Type of task for which format will be set. Can be 
                Single Sentence Classification
                Sentence Pait Classification
    hasLabels - Whether or not the file has labels. When hasLabels is not True, it will
                make dummy labels

    '''
    allData = []
    for line in open(dataPath):
        cols = line.strip("\n").split("\t")

        if hasLabels is True:
            assert len(cols) == 3, "Data not in NER format"
            row = {"uid":cols[0], "label":literal_eval(cols[1]), "sentence":literal_eval(cols[2])}
            assert type(row['label'])==list, "Label should be in list of token labels format in data"
        else:
            row = {"uid":cols[0], "label": ["O"]*len(literal_eval(cols[1])), "sentence":literal_eval(cols[1])}
        assert type(row['sentence'])==list, "Sentence should be in list of token labels format in data"

        allData.append(row)

    return allData


def create_data_ner(data, chunkNumber, tempList, maxSeqLen, tokenizer, labelMap):
    '''
    Function to create data in NER/Sequence Labelling format.
    The tsv format expected by this function is 
    sample['uid] :- unique sample/sentence id
    sample['sentence'] :- list of the sentence tokens for the sentence eg. ['My', 'name', 'is', 'hello']
    sample['label] :- list of corresponding tag for token in sentence ed. ['O', 'O', 'O', 'B-PER']

    Here we won't use the standard data converter as format for NER data 
    is slightly different and required different steps to prepare.

    The '[CLS]' and '[SEP]' also has to be added in label front and end, as they will
    be present in sentence start and end.
    Word piece tokenizer breaks a single word into multiple parts if
    its unknown, we need to add 'X' in label for extra pieces

    '''
    name = 'ner_{}.json'.format(str(chunkNumber))

    with open(name, 'w') as wf:
        with tqdm(total = len(data), position = chunkNumber) as progress:  
            for idx, sample in enumerate(data):
                ids = sample['uid']
                tempTokens = ['[CLS]']
                tempLabels = ['[CLS]']
                for word, label in zip(sample['sentence'], sample['label']):
                    tokens = tokenizer.tokenize(word)
                    for m, token in enumerate(tokens):
                        tempTokens.append(token)
                        # #only first piece would be marked with label
                        # if m==0:
                        #     tempLabels.append(label)
                        # else: 
                        #     tempLabels.append('X')
                        # temp label will append the same label for all the pieces of a word
                        tempLabels.append(label)
                # adding [SEP] at end
                tempTokens.append('[SEP]')
                tempLabels.append('[SEP]')
                out = tokenizer.encode_plus(text = tempTokens, add_special_tokens=False,
                                        padding='max_length', 
                                        truncation=True,
                                        max_length = maxSeqLen)
                # print("len vocab: ", len(tokenizer.get_vocab()))  #30522 
                typeIds = None
                inputMask = None
                tokenIds = out['input_ids']
                if 'token_type_ids' in out.keys():
                    typeIds = out['token_type_ids']
                if 'attention_mask' in out.keys():
                    inputMask = out['attention_mask']

                # doi voi SRL data
                tempLabelsEnc = pad_sequences([ [labelMap[l] for l in tempLabels] ], 
                                    maxlen=maxSeqLen, value=labelMap["O"], padding="post",
                                    dtype="long", truncating="post").tolist()[0]
               
                assert len(tempLabelsEnc) == len(tokenIds), "mismatch between processed tokens and labels"
                features = {
                'uid': ids,
                'label': tempLabelsEnc,
                'token_id': tokenIds,
                'type_id': typeIds,
                'mask': inputMask}

                wf.write('{}\n'.format(json.dumps(features)))  
                progress.update(1)  
        tempList.append(name)

def create_data_multithreaded(data, wrtPath, tokenizer, taskObj, taskName, maxSeqLen, multithreaded):
    '''
    This function uses multi-processing to create the data in the required format
    for base models as per the task. Utilizing multiple Cores help in processing
    huge data with speed
    '''
    man = mp.Manager()

    # shared list to store all temp files written by processes
    tempFilesList = man.list()
    
    if multithreaded:
        # numProcess = mp.cpu_count() - 1
        numProcess = 1

    '''
    Dividing the entire data into chunks which can be sent to different processes.
    Each process will write its chunk into a file. 
    After all processes are done writing, we will combine all the files into one
    '''

    labelMap = taskObj.labelMap[taskName]

    chunkSize = int(len(data) / (numProcess))
    print('Data Size: ', len(data))
    print('number of threads: ', numProcess)

    processes = []
    for i in range(numProcess):
        dataChunk = data[chunkSize*i : chunkSize*(i+1)]

        p = mp.Process(target = create_data_ner, args = (dataChunk, i, tempFilesList, maxSeqLen, tokenizer, labelMap))
        
        p.start()
        processes.append(p)
        
    for pr in processes:
        pr.join()
    
    # combining the files written by multiple processes into a single final file
    with open(wrtPath, 'w') as f:
        for file in tempFilesList:
            with open(file, 'r') as r:
                for line in r:
                    sample =  json.loads(line)
                    f.write('{}\n'.format(json.dumps(sample)))
            os.remove(file)
def main():
    
    # taking in arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--task_file', type=str, default="tasks_file.yml")
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--do_lower_case', type=bool, default=True)
    parser.add_argument('--max_seq_len', type=int, default = 85,
                        help = "max sequence length for making data for model")
    parser.add_argument('--has_labels', type=bool, default=True,
                        help = "If labels are not present in file then False. \
                            To be used when preparing data for inference ")
    parser.add_argument('--multithreaded', type=bool, default=True, \
                        help = "use multiple threads for processing data with speed")
    args = parser.parse_args()
    
    # print current directory
    print('Current Directory: ', os.getcwd() + args.data_dir)
    tasks = TasksParam(args.task_file)
    print('task object created from task file...')
    assert os.path.exists(args.data_dir), "data dir doesnt exist"

    modelName = tasks.modelType.name.lower()
    _, _, tokenizerClass, defaultName = NLP_MODELS['bert']
    configName = tasks.modelConfig
    if configName is None:
        configName = defaultName
    
    #making tokenizer for model
    tokenizer = tokenizerClass.from_pretrained(configName)
    print('{} model tokenizer loaded for config {}'.format(modelName, configName))
    # dataPath = os.path.join(args.data_dir, '{}_prepared_data'.format(configName))
    dataPath = os.path.join(args.data_dir, 'modify_label_prepared_data')
    if not os.path.exists(dataPath):
        os.makedirs(dataPath)

    for taskId, taskName in tasks.taskIdNameMap.items():
        for file in tasks.fileNamesMap[taskName]:
            print('Loading raw data for task {} from {}'.format(taskName, os.path.join(args.data_dir, file)))
            rows = load_data(os.path.join(args.data_dir, file), hasLabels = args.has_labels)
            #wrtFile = os.path.join(dataPath, '{}.json'.format(file.split('.')[0]))
            wrtFile = os.path.join(dataPath, '{}.json'.format(file.lower().replace('.tsv', '')))
            print('Processing Started...')
            create_data_multithreaded(rows, wrtFile, tokenizer, tasks, taskName,
                                    args.max_seq_len, args.multithreaded)
            print('Data Processing done for {}. File saved at {}'.format(taskName, wrtFile))
           
            
            
if __name__ == "__main__":
    main()