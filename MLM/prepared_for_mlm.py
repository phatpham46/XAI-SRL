
import os
import re
import torch
import spacy
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from sklearn.model_selection import train_test_split
from mlm_utils.preprocess_functions import check_data_dir
from mlm_utils.preprocess_functions import get_pos_tag_word
from mlm_utils.model_utils import MLM_IGNORE_LABEL_IDX, VOCAB_SIZE, BATCH_SIZE, EPOCHS, MAX_SEQ_LEN, BERT_PRETRAIN_MODEL, NLP, TOKENIZER


MAX_SEQ_LEN = 85
NUMBER_WORKERS = 5 
BERT_PRETRAINED_MODEL = 'dmis-lab/biobert-base-cased-v1.2'

VOCAB_SIZE = 28996 
wwm_probability = 0.1

   
def get_files(dir: str) -> list:
    '''Function to get the list of files in a given folder'''
    
    files = []
    for path in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, path)):
            files.append(path)
    return files

def decode_token(input_ids: list, skip_special_tokens=False) -> str:
    ''' Funciton to decode the token to text
    '''
    
    return TOKENIZER.decode(input_ids, skip_special_tokens=skip_special_tokens, return_offsets_mapping=True)

def encode_text(text: str) -> dict:
    ''' Function to encode the text '''
    return TOKENIZER.encode_plus(
                    text,
                    max_length=MAX_SEQ_LEN,
                    padding='max_length', 
                    truncation=True,  
                    add_special_tokens = True,
                    return_tensors="pt",  
                    return_attention_mask = True,
                    return_offsets_mapping=True  
                )

def convert_csv_to_tsv(readDir: str, writeDir: str) -> None:
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
                tokens = NLP(words[i])
                
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
 

def get_tokens_for_words(words: list, input_ids: list, offsets: list) -> dict:
    '''
    Input:
        words = ['The', 'capital', 'of', 'the', 'France', 'is', 'Paris', '.']
        input_ids = [101, 1109, 3007, 1104, 2605, 1110, 3000, 119, 102]
        offsets = [(None, None), (0, 3), (4, 11), (12, 14), (15, 17), (18, 23), (24, 25), (None, None)]
    Output: 
        word_dict = {'The': [tensor(101), tensor(170), tensor(170)], 'capital': [tensor(1109), tensor(3007)], ....}
    '''
    

    word_dict = {}
    current_word = ''
    token_list = []
    sentence = ' '.join(words)
    except_tokens = [TOKENIZER.cls_token_id, TOKENIZER.sep_token_id, TOKENIZER.pad_token_id, TOKENIZER.unk_token_id]
    for j, (token, offset) in enumerate(zip(input_ids[0], offsets)):
       
        if offset[0] is not None:  # If the token is associated with a word in the original text
            
            start, end = offset
            original_word = sentence[start:end]
            # original_word = TOKENIZER.decode(input_ids[j], skip_special_tokens=True)
            if j > 0 and offset[0] == offsets[j - 1][1]:  # Check if the current word should be concatenated
                current_word += original_word
                token_list.append(token)
            else:
                if current_word:
                    word_dict[current_word] = [token_list]
                current_word = original_word
                token_list = [token]

    # Add the last word
    if current_word:
        word_dict[current_word] = [token_list]

    return {word: [item for sublist in tokens 
                            for item in sublist 
                                if item not in except_tokens ]  for (word, tokens) in word_dict.items() }


def mask_content_words(ids: torch.Tensor, word_dict: dict) -> tuple:
    '''
    input:
        ids: sample_id 
        word_dict: {'The': [tensor(170)], 'capital': [tensor(1109), tensor(3007)], ....}
    output:
        masked_sentences: [tensor([  101,  1103, 175, 10555,  103,   103,   103,   102])]
        label_ids: [tensor(6468)], [tensor(1568), tensor(13892)]
    '''
    labels = []
    masked_sentences = []
    except_tokens = [TOKENIZER.cls_token_id, TOKENIZER.sep_token_id, TOKENIZER.pad_token_id, TOKENIZER.unk_token_id]
  
    for (key, value) in word_dict.items():
        masked_ids = ids.clone() # torch.Size([1, 85])
        origin_sample = decode_token(masked_ids[0], skip_special_tokens=True)
        if get_pos_tag_word(key, origin_sample).get(key) in ['NOUN', 'VERB', 'ADJ', 'ADV']:
            
            for i in range(len(masked_ids[0]) - len(value) + 1):
                masked_id = masked_ids.clone()
                label = torch.full_like(masked_id, fill_value=-100)
                masked_indice = torch.full_like(masked_id, fill_value=0)
            
                if torch.equal(torch.as_tensor(masked_id[0][i:i+len(value)]).clone().detach(), torch.as_tensor(value).clone().detach()):
                    masked_id[0][i:i+len(value)]= TOKENIZER.mask_token_id
                    #print("MASKED ID: ", masked_id)
                    masked_indice[masked_id == TOKENIZER.mask_token_id] = 1
                    #print("MASKED INDICE: ", masked_indice)
                    
                    
                    for idx, mask in enumerate(masked_indice[0]):
                        if mask == 1:
                            label[0][idx] = ids[0][idx]
                            
                    masked_sentences.append(masked_id)
                    labels.append(label)
            
    return masked_sentences, labels

def masking_sentence_word(words: list, input_ids: torch.tensor, offsets: list) -> tuple:
    '''
    Input: 
        words = ['The', 'capital', 'of', 'France', 'is', 'Paris', '.']
    Output: 
        masked_sentences = [tensor([  101,  1103, 175, 10555,  1110,   103,   119,   102]), [  101,  1103,  2364, 10555,  103,   103,   119,   102]] 
        label_ids = [tensor(6468)], [tensor(1568), tensor(13892)]
    '''
    # get a list of token for the word
    word_dict = get_tokens_for_words(words, input_ids, offsets)
   
    # masked the tokens if the word is the content word
    masked_sentences, label_ids = mask_content_words(input_ids, word_dict)
    
    return masked_sentences, label_ids


def get_word_list(text: str) -> list:
    '''Function to get the list of words from a given text using spacy'''
    
    doc = NLP(text)
    word_lst = [word.text for word in [token for token in doc]]
    return word_lst

def tokenize_csv_to_json(dataDir: str, wriDir: str) -> None:
    '''
    Tokenize_csv_to_json('./interim/', './mlm_output/')
    Function to create data in MLM format.
    Input file: csv with columns ['id', 'source ,'text', 'arguments']
    Output file: json with columns ['uid', 'token_id', 'mask', 'pos']
    
    '''
    
    # check if the data directory exists
    check_data_dir(dataDir, auto_create=False)
    check_data_dir(wriDir, auto_create=True)
    
    
    files = get_files(dataDir)
    for file in files:
        
        features = []
        writeFile = os.path.join(wriDir, 'mlm_{}.csv'.format(file.split('.')[0]))
        
        data = pd.read_csv(os.path.join(dataDir, file))
        print("Processing file: ", file)
    
        with tqdm(total=len(data['text'])) as pbar:
            for sample in data['text']:    
            
                # Get word list from sample
                word_lst = get_word_list(sample)
                
                # Encode the sentence
                tokenized_sentence = encode_text(' '.join(word_lst))
                
                # Mask the content words
                masked_sens, label_ids = masking_sentence_word(
                    word_lst, 
                    tokenized_sentence['input_ids'], 
                    tokenized_sentence['offset_mapping'][0]
                    )
                # Create a feature for each masked sentence
                for mask, label in zip(masked_sens, label_ids):
                    assert len(mask[0]) == MAX_SEQ_LEN, "Mismatch between processed tokens and labels"
                    
                    feature = {
                        'token_id': mask[0].numpy().tolist(), 
                        'attention_mask': tokenized_sentence['attention_mask'][0].numpy().tolist(),  
                        'token_type_ids': tokenized_sentence['token_type_ids'][0].numpy().tolist(), 
                        'labels': label[0].numpy().tolist()}
                
                    features.append(feature)
                
                # Update the progress bar
                pbar.update(1)
            
            
        # Write to a CSV file
        df_feature = pd.DataFrame(features)
        df_feature.to_csv(writeFile, index = False)
           

def data_split(dataDir: str, wriDir: str) -> tuple:
    
    '''
    data_split('mlm_output', 'mlm_prepared_data')
    Function to split data into train, dev, test (60, 20, 20) and merge to json files.
    '''
    check_data_dir(dataDir, auto_create=False)
    check_data_dir(wriDir, auto_create=True)   
    
    files = get_files(dataDir)
    train_df = pd.DataFrame()
    dev_df = pd.DataFrame()
    test_df = pd.DataFrame()
     
    for file in files:
       
        data = pd.read_csv(os.path.join(dataDir, file))    
        train, testt = train_test_split(data, test_size=0.4)
        dev, test = train_test_split(testt, test_size=0.5)
        
        # concatenate the data
        train_df = pd.concat([train_df, train], ignore_index=True)
        dev_df = pd.concat([dev_df, dev], ignore_index=True)
        test_df = pd.concat([test_df, test], ignore_index=True)
        
        print("Processing file: ", file)
    
    train_df.to_json(os.path.join(wriDir, 'train_mlm.json'), orient='records', lines=True)
    dev_df.to_json(os.path.join(wriDir, 'dev_mlm.json'), orient='records', lines=True)
    test_df.to_json(os.path.join(wriDir, 'test_mlm.json'), orient='records', lines=True)
    
    return train_df, dev_df, test_df


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


# def create_training_data(dataDir, wriDir):
#     output_dir = Path(wriDir)
#     output_dir.mkdir(exist_ok=True, parents=True)
#     vocab_list = list(TOKENIZER.vocab.keys())    
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
    
    # tokenize_csv_to_json('./interim/', './mlm_output_3/')
    
    data_split('./mlm_output_3/', './mlm_prepared_data_3/')
    
if __name__ == "__main__":
    main() 



