from transformers import AutoTokenizer, BertForMaskedLM
import torch
import copy
import random

tokenizer = AutoTokenizer.from_pretrained("dmis-lab/biobert-v1.1")
model = BertForMaskedLM.from_pretrained("dmis-lab/biobert-v1.1")


MAX_SEQ_LEN = 70

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
    
    masked_idx = random.sample(range(len(words)), 18)
    print("masked_idx", masked_idx)
    
    # create 10 sentences with 10 masked tokens
    
    sen_10 = []
    label_10 = []
    labels = [-100] * MAX_SEQ_LEN 
  
    for i in range(15): 
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

def main():
    import re
    text = 'In addition, deletion of the distal tor box (box1) abolished torC induction whereas the presence of a DNA fragment starting three bases upstream from box1 suffices for normal torC expression.'
    words = re.split(r'([.,;\s])', text)    
    words = [word for word in words if word != ' ' and word != ''] 
    print(words)
    print("masked sentence \n")
    print(masking_sentence_word(words, tokenizer)[0])

if __name__ == "__main__":
    main()