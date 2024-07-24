import spacy
import torch
from enum import IntEnum
from transformers import BertConfig, BertModel, BertTokenizerFast
from SRL.loss import *
from utils.tranform_functions import pasbio_srl_to_tsv, get_embedding, get_embedding_finetuned, convert_csv_to_txt
from utils.eval_metrics import *
from multiprocessing import cpu_count

# Load the English language model
NLP = spacy.load("en_core_web_sm")

MAX_SEQ_LEN = 85

POS_TAG_MAPPING = {"NOUN":1, "VERB":2, "ADJ":3, "ADV":4}

BERT_PRETRAIN_MODEL = "dmis-lab/biobert-base-cased-v1.2"

bioBertTokenizer = BertTokenizerFast.from_pretrained(BERT_PRETRAIN_MODEL, do_lower_case=True, truncation=True)

NLP_MODELS = {
    "bert": (BertConfig, BertModel, bioBertTokenizer, BERT_PRETRAIN_MODEL), 
}

def count_num_cpu_gpu():
    if torch.cuda.is_available():
        num_gpu_cores = torch.cuda.device_count()
        num_cpu_cores = (cpu_count() // num_gpu_cores // 2) - 1
    else:
        num_gpu_cores = 0
        num_cpu_cores = (cpu_count() // 2) - 1
    return num_cpu_cores, num_gpu_cores
NUM_CPU, NUM_GPU = count_num_cpu_gpu()


TRANSFORM_FUNCS = {
    "pasbio_srl_to_tsv" : pasbio_srl_to_tsv,
    "get_embedding" : get_embedding,
    "get_embedding_finetuned" : get_embedding_finetuned,
    "convert_csv_to_txt": convert_csv_to_txt
}

class ModelType(IntEnum):
    BERT = 1

class TaskType(IntEnum):
    SRL = 3

class LossType(IntEnum):
    CrossEntropyLoss = 0
    SRLLoss = 1
    
METRICS = {
    "classification_accuracy": classification_accuracy,
    "classification_f1_score": classification_f1_score,
    "seqeval_f1_score" : seqeval_f1_score,
    "seqeval_precision" : seqeval_precision,
    "seqeval_recall" : seqeval_recall,
    "snips_f1_score" : snips_f1_score,
    "snips_precision" : snips_precision,
    "snips_recall" : snips_recall,
    "classification_recall" : classification_recall
}

LOSSES = {
    "crossentropyloss" : CrossEntropyLoss,
    "nerloss" : SRLLoss
}
