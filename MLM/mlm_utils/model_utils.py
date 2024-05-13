import spacy
import torch
from transformers import BertTokenizerFast, BertForMaskedLM
from multiprocessing import cpu_count

MLM_IGNORE_LABEL_IDX = -1
VOCAB_SIZE = 28996 
BATCH_SIZE = 32

MAX_SEQ_LEN = 128
BERT_PRETRAIN_MODEL = "dmis-lab/biobert-base-cased-v1.2"
TOKENIZER = BertTokenizerFast.from_pretrained(BERT_PRETRAIN_MODEL, do_lower_case=True)

# Load the English language model
NLP = spacy.load("en_core_web_sm")

POS_TAG_MAPPING = {"NOUN":1, "VERB":2, "ADJ":3, "ADV":4}

def count_num_cpu_gpu():
    if torch.cuda.is_available():
        num_gpu_cores = torch.cuda.device_count()
        num_cpu_cores = (cpu_count() // num_gpu_cores // 2) - 1
    else:
        num_gpu_cores = 0
        num_cpu_cores = (cpu_count() // 2) - 1
    return num_cpu_cores, num_gpu_cores

NUM_CPU, NUM_GPU = count_num_cpu_gpu()

