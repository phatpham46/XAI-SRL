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


def get_pos_tag(text, index):
    # Process the text with spaCy
    doc = nlp(text)

    # Check if the index is within the bounds
    if index < 0 or index >= len(doc):
        return "Index out of bounds"

    # Get the POS tag for the word at the specified index
    pos_tag = doc[index].pos_
    return pos_tag


def get_pos_tags(data):
    pos_tags = []
    for line in data:
        pos_tag = get_pos_tag(line, index)
        line['pos_tag'] = pos_tag
        
def pos_match(textA, textB, index):
    return 1 if get_pos_tag(textA, index) == get_pos_tag(textB, index) else 0

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
                uids = data['id'][idx]   
                
                # Process the tokenized text with spaCy
                # doc = nlp(sample)
                words = re.split(r'([.,;\s])', sample)    
                words = [word for word in words if word != ' ' and word != ''] 
                
                mask_10, labels_10 = masking_sentence_word(words, tokenizer)  # mask là masked_sentence
                # ['A', 'G-to-A', 'transition', 'at', '[MASK]', 'first', 'nucleotide', 'of', 'intron', '2', 'of', '[MASK]', '1', 'abolished', normal', 'splicing', '.']
                
                # pos_tags = [token.pos_ for token in doc]
                for mask, label in zip(mask_10, labels_10):
                   
                    out = tokenizer.encode_plus(text = ' '.join(mask), add_special_tokens=True,
                                    truncation_strategy ='only_first',
                                    max_length = MAX_SEQ_LEN, pad_to_max_length=True) 

                    attention_mask = None
                    tokenIds = out['input_ids']                
                    if 'attention_mask' in out.keys():
                        attention_mask = out['attention_mask']
                            
                    assert len(tokenIds) == MAX_SEQ_LEN, "Mismatch between processed tokens and labels"
                    
                    feature = {'uid':str(uids), 'token_id': tokenIds, 'attention_mask': attention_mask, 'labels': label}
                    
                    wf.write('{}\n'.format(json.dumps(feature))) 
            print("Done file: ", file)
           
def is_in_vocab(token, tokenizer):
    '''
    Function to check if a token is in the vocabulary
    '''
    return 1 if str(token).lower() in tokenizer.vocab.keys() else 0

def masking_sentence_word(words, tokenizer):
    '''
    Function to mask random token in a sentence and return the masked sentence and the corresponding label ids
    '''
    except_tokens = [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]
    
    masked_idx = random.sample(range(len(words)), len(words) - 2)
    print(masked_idx)
    # create 10 sentences with 10 masked tokens
    
    sen_10 = []
    label_10 = []
    labels = [-100] * MAX_SEQ_LEN 
  
    for i in range(len(words)-2): 
        tmp_sen = copy.deepcopy(words)
       
        tmp_label = copy.deepcopy(labels)
        if len(sen_10) < 10:
            masked_token = tmp_sen[masked_idx[i]]
            if (masked_token not in except_tokens) and is_in_vocab(tmp_sen[masked_idx[i]], tokenizer) == 1:
               
                tmp_label[masked_idx[i]] = tokenizer.convert_tokens_to_ids(tmp_sen[masked_idx[i]])
                tmp_sen[masked_idx[i]] = tokenizer.mask_token
                
                sen_10.append(tmp_sen)
                label_10.append(tmp_label)
        else :
            break
       
    return sen_10, label_10
    
def masked_df(df, tokenizer):
    '''
    Function to mask tokens in a dataframe and return the masked dataframe.
    df has columns ['uid', 'token_id', 'mask', 'pos', 'masked_token_id', 'label_id']
    '''
    masked_sentences = []
    labels = []
    for idx, sen in enumerate(df['token_id']):
        masked_sentence, label = masking_sentence_word(sen, tokenizer)

        masked_sentences.append(masked_sentence)
        labels.append(label)
    
    df['masked_token_id'] = masked_sentences
    df['label_id'] = labels
    return df

def data_split(dataDir, wriDir, tokenizer, batch_size=32):
    
    '''
    data_split('mlm_output', 'mlm_prepared_data', tokenizer)
    Function to split data into train, dev, test (60, 20, 20) and merge to json files.
    '''
    files = get_files(dataDir)

    train_df = pd.DataFrame()
    dev_df = pd.DataFrame()
    test_df = pd.DataFrame()
    
    for file in files:
        #f = open(os.path.join(dataDir, file))
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
    
    # We'll take training samples in random order. 
    # train_dataloader  = DataLoader(
    #             train_df,  # The training samples.
    #             sampler = RandomSampler(train_df), # Select batches randomly
    #             batch_size = batch_size # Trains with this batch size.
    #         )
    # validation_dataloader = DataLoader(
    #             dev_df, # The validation samples.
    #             sampler = SequentialSampler(dev_df), # Pull out batches sequentially.
    #             batch_size = batch_size # Evaluate with this batch size.
    #       )
  
    # test_dataloader = DataLoader(
    #             test_df, # The validation samples.
    #             sampler = SequentialSampler(test_df), # Pull out batches sequentially.
    #             batch_size = batch_size # Evaluate with this batch size.
    #         )
    #return train_dataloader, validation_dataloader, test_dataloader
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
    
    data_split('mlm_output', 'mlm_prepared_data', tokenizer)
    
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
