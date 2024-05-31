
import json
import os
import re
import torch
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from mlm_utils.transform_func import get_pos_tag_word, get_word_list, check_data_dir, decode_token, encode_text, get_files, pos_tag_mapping
from mlm_utils.model_utils import MAX_SEQ_LEN, NLP, TOKENIZER

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


def masking_sentence_word(words: list, input_ids: torch.tensor, offsets: list):
    '''
    Input: 
        words = ['The', 'capital', 'of', 'France', 'is', 'Paris', '.']
        input_ids: [tensor([  101,  1103, 175, 10555,  1110,   1113,   119,   102])]
    Output: 
        pos_tag_id: [0, 0, 0, 0, 1, 1, 1, 0] (1: noun, 2: verb, ..)
    '''
    # get a list of token for the word
    word_dict = get_tokens_for_words(words, input_ids, offsets)
    
    # create a list of masked sentence from one input id
    pos_tag_id = []
    list_masked_id = []
    for (key, value) in word_dict.items():
        masked_ids = input_ids.clone() 
        origin_sample = decode_token(masked_ids[0], skip_special_tokens=True)
        if get_pos_tag_word(key, origin_sample).get(key) in ['NOUN', 'VERB', 'ADJ', 'ADV']:
            
            for i in range(len(masked_ids[0]) - len(value) + 1):
                masked_id = masked_ids.clone()
                label = torch.full_like(masked_id, fill_value=0)
                masked_indice = torch.full_like(masked_id, fill_value=0)
            
                if torch.equal(torch.as_tensor(masked_id[0][i:i+len(value)]).clone().detach(), torch.as_tensor(value).clone().detach()):
                    masked_id[0][i:i+len(value)]= TOKENIZER.mask_token_id
                    masked_indice[masked_id == TOKENIZER.mask_token_id] = 1
                    
                    for idx, mask in enumerate(masked_indice[0]):
                        if mask == 1:
                            label[0][idx] = pos_tag_mapping(get_pos_tag_word(key, origin_sample).get(key))
                            
                    pos_tag_id.append(label)
                    list_masked_id.append(masked_id)
    
    return input_ids, pos_tag_id, list_masked_id

def data_preprocessing(dataDir: str, labelDir: str, wriDir: str):
    '''
    data_preprocessing('./interim/', './mlm_output/')
    Function to create data in MLM format.
    Input file: csv with columns ['id', 'source ,'text', 'arguments']
    Output file: json with columns ['uid', 'token_id', 'mask', 'pos']
    
    '''
    
    # check if the data directory exists
    check_data_dir(dataDir, auto_create=False)
    check_data_dir(wriDir, auto_create=True)
    
    files = get_files(dataDir)
    label_files = get_files(labelDir)
    for file, label_file in zip(files, label_files):
        print("Processing file: ", file, "label file ", label_file)
        features = []
        writeFile = os.path.join(wriDir, 'mlm_{}.csv'.format(file.split('.')[0]))
        
        data = pd.read_csv(os.path.join(dataDir, file))
        with open(os.path.join(labelDir, label_file), 'r') as f1: 
            data_label = [json.loads(line) for line in f1]
            
        # Create a dictionary from data_label for quick lookup
        label_dict = {int(item['uid']): item['label'] for item in data_label}

        with tqdm(total=len(data['text'])) as pbar:
            for id, sample in zip(data['id'], data['text']):    
                if id in label_dict:
                    label = label_dict[id]
                
                
                # Get word list from sample
                word_lst = get_word_list(sample)
                
                # Encode the sentence
                tokenized_sentence = encode_text(' '.join(word_lst))
                
                # Mask the content words
                input_id, pos_tag_ids, list_masked_id = masking_sentence_word(
                    word_lst, 
                    tokenized_sentence['input_ids'], 
                    tokenized_sentence['offset_mapping'][0]
                    )
                # Create a feature for each masked sentence
                for pos_tag_id, masked_id in zip(pos_tag_ids, list_masked_id):
                    
                    feature = {
                        'uid': id,  # origin_id
                        'token_id': input_id[0].numpy().tolist(),
                        'mask': tokenized_sentence['attention_mask'][0].numpy().tolist(),  
                        'type_id': tokenized_sentence['token_type_ids'][0].numpy().tolist(),
                        'pos_tag_id': pos_tag_id[0].numpy().tolist(),
                        'masked_id': masked_id[0].numpy().tolist(),
                        'label': label
                       }
                
                    features.append(feature)
                
                # Update the progress bar
                pbar.update(1)
     
        with open(writeFile.replace('csv', 'json'), 'w') as wf:
            for feature in features:
                wf.write('{}\n'.format(json.dumps(feature))) 
            
        
def main():
    
    data_preprocessing('./data_mlm/raw_folder/interim/', './data_mlm/process_folder/coNLL_tsv_json/ner_json/', './data_mlm/process_folder/mlm_output_v2/')
    
if __name__ == "__main__":
    main() 