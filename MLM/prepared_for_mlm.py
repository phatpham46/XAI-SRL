import copy
import os
import re
import numpy as np
import pandas as pd
import json
from ast import literal_eval
from sklearn.model_selection import train_test_split
import spacy
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer 
from pathlib import Path
from tqdm import tqdm, trange
MAX_SEQ_LEN = 85
VOCAB_SIZE = 28996 
wwm_probability = 0.1
import random
# Load the English language model
nlp = spacy.load("en_core_web_sm")
      
def get_files(dir):
    files = []
    for path in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, path)):
            files.append(path)
    return files

def convert_csv_to_tsv(readDir, writeDir):
    '''
    convert_csv_to_tsv('../MLM/interim/', 'coNLL_tsv')
    '''
    def convert_to_ner_format(df, file):
        ner_format = []
        words_tokenized = []
        
        # Extract the predicate from the filename
        base_name = os.path.basename(file)
        predicate = os.path.splitext(base_name)[0].split('_')[0]
        
        # convert arguments to dictionary
        df['arguments'] = df['arguments'].apply(lambda x: eval(x))
        for index, row in df.iterrows():
            text = str(row['text']).lower()
            arguments = row['arguments']
            
            # Tokenize the text into words while preserving selected punctuation
            words = re.split(r'([.,;\s])', text)    
            words = [word for word in words if word != ' ' and word != ''] 

            ner_tags = ['O'] * len(words)
            
            for i in range(len(words)):
                tokens = nlp(words[i])
                
                if (tokens[0].lemma_.lower() == predicate):
                    words[i] = '#' + words[i]
                    ner_tags[i] = 'B-V'
                    break
            
            for arg_id, arg_text in arguments.items():           
                arg_text = arg_text.lower()
                
                # Tokenize the argument into words
                arg_words = re.split(r'([.,;\s])', arg_text)    
                arg_words = [word for word in arg_words if word != ' ' and word != '']
                
                # Iterate through the words in the sentence
                for i in range(len(words) - len(arg_words) + 1):
                    if words[i:i+len(arg_words)] == arg_words:
                        # Assign a label based on the argument key
                        for j in range(len(arg_words)):
                            if j == 0:
                                ner_tags[i+j] = f'B-A{arg_id}'
                            else:
                                ner_tags[i+j] = f'I-A{arg_id}'
            ner_format.append(ner_tags)
            words_tokenized.append(words)
        return ner_format, words_tokenized
    
    files = get_files(readDir)
            
    for file in files:
        data_df = pd.read_csv(os.path.join(readDir, file), sep=',', header=0)
        ner_format = convert_to_ner_format(data_df, file)
        labelNer = ner_format[0]
        uid = data_df['id']                      
        text = ner_format[1]
        
        nerW = open(os.path.join(writeDir, 'ner_{}.tsv'.format(file.split('.')[0])), 'w')
        for i in range(len(uid)):
            nerW.write("{}\t{}\t{}\n".format(uid[i], labelNer[i], text[i]))
    nerW.close()
 
def tokenize_csv_to_json(dataDir, wriDir, tokenizer):
    '''
    Tokenize_csv_to_json('./interim/', './mlm_output/', tokenizer)
    Function to create data in MLM format.
    Input file: csv with columns ['id', 'source ,'text', 'arguments']
    Output file: json with columns ['uid', 'token_id', 'mask', 'pos']
    
    '''
    
    # Read train file
    files = get_files(dataDir)
    
    for file in files:
        writefile = os.path.join(wriDir, 'mlm_{}.json'.format(file.split('.')[0]))
        with open(writefile, 'w') as wf:
            data = pd.read_csv(os.path.join(dataDir, file))
            print("Processing file: ", file)
            
            for idx, sample in enumerate(data['text']) :    
                count_masked_sens = 0
                uids = data['id'][idx]   
                
                # Get the POS tag for each word in the text
                doc = nlp(sample)

                # Get the POS tag for each word in the text
                pos_tags = [token.pos_ for token in doc]
                
                # words = re.split(r'([.,;\s])', sample)    
                # words = [word for word in words if word != ' ' and word != ''] 
                words = [token for token in doc]
                mask_sens, labels_sens = masking_sentence_word(words, tokenizer, pos_tags)  # mask là masked_sentence
                # ['A', 'G-to-A', 'transition', 'at', '[MASK]', 'first', 'nucleotide', 'of', 'intron', '2', 'of', 'index', '1', 'abolished', normal', 'splicing', '.']
                
                for mask, label in zip(mask_sens, labels_sens):
                    
                    out = tokenizer.encode_plus(text = ' '.join(mask), add_special_tokens=True,
                                    truncation_strategy ='only_first',
                                    max_length = MAX_SEQ_LEN, padding='max_length') 

                    attention_mask = None
                    tokenIds = out['input_ids']                
                    if 'attention_mask' in out.keys():
                        attention_mask = out['attention_mask']
                            
                    assert len(tokenIds) == MAX_SEQ_LEN, "Mismatch between processed tokens and labels"
                    
                    #feature = {'uid':str(uids), 'token_id': tokenIds, 'attention_mask': attention_mask, 'labels': label}
                    feature = {'uid':str(uids)+str('_')+str(count_masked_sens), 'token_id': tokenIds, 'attention_mask': attention_mask, 'labels': label}
                    count_masked_sens += 1 
                    wf.write('{}\n'.format(json.dumps(feature))) 
            print("Done file: ", file)
           
def is_in_vocab(token, tokenizer):
    '''
    Function to check if a token is in the vocabulary
    '''
    return 1 if str(token).lower() in tokenizer.vocab.keys() else 0

def masking_sentence_word(words, tokenizer, pos_tags):
    '''
    Function to mask each token (only token in vocab and content word) in a sentence and return the masked sentence and the corresponding label ids
    
    input: words: the full token of original sentence
           tokenizer: the tokenizer object
           
    output: masked_sentences: a list of masked sentences from the original sentence
            label_ids: a list of label ids corresponding to the masked sentences
    '''
    except_tokens = [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]
    
    # create a list to store the masked sentences and the corresponding label ids
    masked_sens = []
    labels_masked_sens = []
    labels = [-100] * MAX_SEQ_LEN 
  
    for i in range(len(words)): 
        tmp_sen = [str(word) for word in words]

        tmp_label = copy.deepcopy(labels)
        if (tmp_sen[i] not in except_tokens) and is_in_vocab(tmp_sen[i], tokenizer) == 1 and pos_tags[i] in ['NOUN', 'VERB', 'ADJ', 'ADV']:
            
            tmp_label[i+1] = tokenizer.convert_tokens_to_ids(tmp_sen[i])
            tmp_sen[i] = tokenizer.mask_token
            
            masked_sens.append(tmp_sen)
            labels_masked_sens.append(tmp_label)
       
    return masked_sens, labels_masked_sens

def data_split(dataDir, wriDir):
    
    '''
    data_split('mlm_output', 'mlm_prepared_data', tokenizer)
    Function to split data into train, dev, test (60, 20, 20) and merge to json files.
    '''
    files = get_files(dataDir)

    train_df = pd.DataFrame()
    dev_df = pd.DataFrame()
    test_df = pd.DataFrame()
    
    for file in files:
       
        with open(os.path.join(dataDir, file)) as f:
            json_data = pd.read_json(f, lines=True)
            
        train, testt = train_test_split(json_data, test_size=0.4)
        dev, test = train_test_split(testt, test_size=0.5)
        
        
        train_df = pd.concat([train_df, train], ignore_index=True)
        dev_df = pd.concat([dev_df, dev], ignore_index=True)
        test_df = pd.concat([test_df, test], ignore_index=True)
        
        print("Processing file: ", file)
    
    
    train_df.to_json(os.path.join(wriDir, 'train_mlm.json'), orient='records', lines=True)
    dev_df.to_json(os.path.join(wriDir, 'dev_mlm.json'), orient='records', lines=True)
    test_df.to_json(os.path.join(wriDir, 'test_mlm.json'), orient='records', lines=True)
    
    return train_df, dev_df, test_df

from multiprocessing import Pool
EPOCHS = 5 
NUMBER_WORKERS = 5 
BERT_PRETRAINED_MODEL = 'dmis-lab/biobert-base-cased-v1.2'

# def create_training_file(docs, epoch_num, output_dir):
#     epoch_filename = output_dir / f"{BERT_PRETRAINED_MODEL}_epoch_{epoch_num}.json"
#     num_instances = 0
#     with epoch_filename.open('w') as epoch_file:
#         for doc_idx in trange(len(docs), desc="Document"):
#             doc_instances = docs[doc_idx]
#             doc_instances = [json.dumps(instance) for instance in doc_instances]
#             for instance in doc_instances:
#                 epoch_file.write(instance + '\n')
#                 num_instances += 1
#     metrics_file = output_dir / f"{BERT_PRETRAINED_MODEL}_epoch_{epoch_num}_metrics.json"


# def create_training_data(dataDir, wriDir, tokenizer):
#     output_dir = Path(wriDir)
#     output_dir.mkdir(exist_ok=True, parents=True)
#     vocab_list = list(tokenizer.vocab.keys())    
#     f = open(dataDir + '/train_mlm.json')
#     docs = json.load(f)
#     f.close()
#     if NUMBER_WORKERS > 1:
#         writer_workers = Pool(min(NUMBER_WORKERS, EPOCHS))
#         arguments = [(docs, idx, output_dir) for idx in range(EPOCHS)]
#         writer_workers.starmap(create_training_file, arguments)
#     else:
#         for epoch in trange(EPOCHS, desc="Epoch"):
#             create_training_file(docs, epoch, output_dir)
            
def main():
    tokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
    #tokenize_csv_to_json('./interim/', './mlm_output/', tokenizer)
    
    data_split('./mlm_output/', './mlm_prepared_data/')
    
if __name__ == "__main__":
    main() 

# là giờ lên colab train hả 
# Chia train test ngay từ đầu         
# 1. Hàm xử lí từng câu(duyệt qua từng token trong câu) - for để mask từng token 
# 
# 2. Hàm xử lí file train(duyệt qua từng câu trong file) => return df
#
# 3. Hàm train truyền vào df, return model
#
# data gom: id, token_id, attention_mask
# df gom: id, masked_token_id, label 

'''
text = "Who was Jim Paterson ? Jim Paterson is a doctor".lower()
inputs  =  tokenizer.encode_plus(text,  return_tensors="pt", add_special_tokens = True, truncation=True, pad_to_max_length = True,
                                         return_attention_mask = True,  max_length=64)
input_ids = inputs['input_ids']
labels  = copy.deepcopy(input_ids) #this is the part I changed
input_ids[0][7] = tokenizer.mask_token_id
labels[input_ids != tokenizer.mask_token_id] = -100 

loss, scores = model(input_ids = input_ids, attention_mask = inputs['attention_mask'] , token_type_ids=inputs['token_type_ids'] , labels=labels)
print('loss',loss)


["DET", "NOUN", "PUNCT", "ADP", "PUNCT", "NOUN", "NOUN", "ADP", "DET", "ADJ", "NOUN", "ADP", "NOUN", "NUM", "ADP", "NOUN", "NUM", "VERB", "ADJ", "NOUN", "PUNCT"]
21
A G-to-A transition at the first nucleotide of intron 2 of patient 1 abolished normal splicing.
'''
# df_train: [input_ids, attention_mask, label]
# file data train bự:
# input_ids, attention_mask, label
