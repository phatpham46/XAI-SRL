from enum import IntEnum

from transformers import BertConfig, BertModel, BertTokenizer
from SRL.loss import *
from utils.tranform_functions import *
from utils.eval_metrics import *

bioBertTokenizer = BertTokenizer.from_pretrained('dmis-lab/biobert-base-cased-v1.2', do_lower_case=True,truncation=True)

MAX_SEQ_LEN = 50

NLP_MODELS = {
    "bert": (BertConfig, BertModel, bioBertTokenizer, 'dmis-lab/biobert-base-cased-v1.2'), # 'bert-base-uncased'
}

TRANSFORM_FUNCS = {
    "coNLL_ner_pos_to_tsv" : coNLL_ner_pos_to_tsv,
    "bio_ner_to_tsv" : bio_ner_to_tsv,
    "get_embedding" : get_embedding,
    "get_embedding_finetuned" : get_embedding_finetuned,
    "convert_csv_to_txt":convert_csv_to_txt
}

class ModelType(IntEnum):
    BERT = 1

class TaskType(IntEnum):
    NER = 3

class LossType(IntEnum):
    CrossEntropyLoss = 0
    NERLoss = 1
    
METRICS = {
    "classification_accuracy": classification_accuracy,
    "classification_f1_score": classification_f1_score,
    "seqeval_f1_score" : seqeval_f1_score,
    "seqeval_precision" : seqeval_precision,
    "seqeval_recall" : seqeval_recall,
    "snips_f1_score" : snips_f1_score,
    "snips_precision" : snips_precision,
    "snips_recall" : snips_recall,
    "classification_recall" : classification_recall,
    "brier_score" : brier_score_for_batch,
}

LOSSES = {
    "crossentropyloss" : CrossEntropyLoss,
    "nerloss" : NERLoss
}
