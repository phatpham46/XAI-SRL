import spacy
from transformers import BertTokenizerFast, BertForMaskedLM
from mlm_utils.utils_mlm import count_num_cpu_gpu

MLM_IGNORE_LABEL_IDX = -1
VOCAB_SIZE = 28996 
BATCH_SIZE = 32
EPOCHS = 2
MAX_SEQ_LEN = 85
BERT_PRETRAIN_MODEL = "dmis-lab/biobert-base-cased-v1.2"
NUM_CPU, NUM_GPU = count_num_cpu_gpu()
BIOBERT_MODEL = BertForMaskedLM.from_pretrained(BERT_PRETRAIN_MODEL)
TOKENIZER = BertTokenizerFast.from_pretrained(BERT_PRETRAIN_MODEL, do_lower_case=True)
# Load the English language model
NLP = spacy.load("en_core_web_sm")



