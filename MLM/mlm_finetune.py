from argparse import ArgumentParser
import math
from pathlib import Path
import time
import torch
import logging
import json
import random
import numpy as np
import pandas as pd
from collections import namedtuple, defaultdict
from tempfile import TemporaryDirectory
from sklearn.metrics import accuracy_score
from babel.dates import format_time
import torch.nn as nn
import torch
import sys
from scipy.special import softmax

sys.path.insert(1, '../')

import sys
# sys.path.insert(1, '/content/SRL-for-BioBERT')
from embedding import read_data
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler 
from sklearn.multioutput import MultiOutputClassifier
from tqdm import tqdm
# from bert_mlm_finetune import BertForMLMPreTraining 
from transformers import BertTokenizer, BertConfig, AdamW, get_linear_schedule_with_warmup, BertForMaskedLM 
from utils_mlm import count_num_cpu_gpu
from prepared_for_mlm import data_split
import spacy
import pathlib
import tensorflow as tf
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
tb = SummaryWriter()

nlp = spacy.load("en_core_web_sm")
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.2", do_lower_case=True)

MLM_IGNORE_LABEL_IDX = -1
VOCAB_SIZE = 28996 
BATCH_SIZE = 32
EPOCHS = 10
MAX_SEQ_LEN = 85
NUM_CPU = count_num_cpu_gpu()[0]

class PregeneratedDataset(Dataset):
    def __init__(self, training_path, file_name, tokenizer, reduce_memory=False):
        self.vocab = tokenizer.vocab
        self.tokenizer = tokenizer
        # self.epoch = epoch
        # self.data_epoch = epoch % num_data_epochs
        
        # train_file = training_path / "train_mlm.json"
        # assert train_file.is_file() 
        # data_list = []
        # with open(train_file) as f:
        #     for line in f:
        #         data = json.loads(line)
        #         data_list.append(data)
        # # num_samples = metrics['num_training_examples']
        # train_df = pd.DataFrame(data_list)
        
        train_df = self.read_df(training_path, file_name)

        num_samples = len(train_df)
        seq_len = MAX_SEQ_LEN
        
        self.temp_dir = None
        self.working_dir = None
        if reduce_memory:
            print("reduce memory")
            self.temp_dir = TemporaryDirectory()
            self.working_dir = Path(self.temp_dir.name)
            self.input_ids = np.memmap(filename=self.working_dir/'input_ids.memmap',
                                  mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
            self.input_masks = np.memmap(filename=self.working_dir/'input_masks.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool)
            self.lm_label_ids = np.memmap(filename=self.working_dir/'lm_label_ids.memmap',
                                     shape=(num_samples, seq_len), mode='w+', dtype=np.int32)
            
        else:
            self.input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            self.input_masks = np.zeros(shape=(num_samples, seq_len), dtype=np.bool)
            self.lm_label_ids = np.full(shape=(num_samples, seq_len), dtype=np.int32, fill_value=MLM_IGNORE_LABEL_IDX)
        
        
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = train_df['token_id']
        self.input_masks = train_df['attention_mask']
        self.lm_label_ids = train_df['labels']
        
    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return (torch.tensor(np.array(self.input_ids[item]), dtype = torch.int64),
                torch.tensor(np.array(self.input_masks[item]), dtype = torch.int64),
                torch.tensor(np.array(self.lm_label_ids[item]), dtype = torch.int64))

    def read_df(self, training_path, file_name):
        train_file = training_path / file_name
        print("train file: ", train_file)
        # assert train_file.is_file() 
        data_list = []
        with open(train_file) as f:
            for line in f:
                data = json.loads(line)
                data_list.append(data)
        
        df = pd.DataFrame(data_list)
        return df
def get_pos_tag(text, index):
    # Process the text with spaCy
    doc = nlp(text)

    # Check if the index is within the bounds
    if index < 0 or index >= len(doc):
       return "Index out of bounds"

    # Get the POS tag for the word at the specified index
    pos_tag = doc[index].pos_
    return pos_tag


def is_POS_match(logits, input_ids, lm_label_ids):
    '''
    Function to check if the POS tag of the masked token in the logits is the same as the POS tag of the masked token in the original text.
    Note: This function assumes that the logits are of shape # ([85, 28996]) 
    lm_label_ids: shape (batch_size, sequence_length)
    '''
    
    origin_input_id = input_ids.clone() # Origin input id:  torch.Size([85])
   
    # Find the index of the masked token from lm_label_ids
    masked_idx = torch.where(lm_label_ids != -100)[0]
    masked_idx_input = torch.where(input_ids == tokenizer.mask_token_id)[0]
   
    origin_input_id[masked_idx_input] = lm_label_ids[masked_idx] 
    
    # get pos tag of origin text
    text = tokenizer.decode(input_ids)
    origin_text = tokenizer.decode(origin_input_id) 
    print("ORIGIN TEXT: ", origin_text)
    pos_tag_origin = get_pos_tag(origin_text, masked_idx_input)
    print("POS TAG ORIGIN: ", pos_tag_origin)
   
    # Extract the logits for the masked position
    masked_logits = logits[0, masked_idx]
    print("MASKED LOGITS: ", masked_logits) # torch.Size([28996])
    # Apply softmax to get probabilities
    probabilities = torch.nn.functional.softmax(masked_logits, dim=-1)

    # Get the token with the highest probability (predicted token)
    predicted_token_id = torch.argmax(probabilities).item()
    predicted_token = tokenizer.convert_ids_to_tokens([predicted_token_id])[0]

    # Replace the [MASK] token with the predicted token in the original text
    result_text = text.replace('[MASK]', predicted_token) 
    print("RESULT TEXT: ", result_text)
    
    # get pos tag of logits
    logits_tag = get_pos_tag(result_text, masked_idx)
    print("LOGITS TAGS: ", logits_tag)
    return pos_tag_origin == logits_tag    

def custom_loss(input_ids, logits, labels):
  
    # Cross-entropy term
    
    cross_entropy_term = F.cross_entropy(logits, labels, reduction='none')
    print("Cross entropy term shape: ", cross_entropy_term.shape)      ##Cross entropy term shape:  torch.Size([2720])
    logits_shape = (32, 85, VOCAB_SIZE)
    logits_tensor = logits.view(*logits_shape)
    
    labels_shape = (32, 85)
    labels_tensor  = labels.view(*labels_shape)
    
    matching_term_lst = []
    # Custom matching term
    for logit, input_id, label in zip(logits_tensor, input_ids, labels_tensor):
       
        matching_term = is_POS_match(logits=logit, input_ids=input_id, lm_label_ids=label)
        print("Matching term: ", matching_term) 
        matching_term_lst.append(matching_term) 
    # hay mình thử sửa cái POS cho cái batch luôn kh, tại cái logit với cái label truyền vô là cái batch á
    # mà t sợ nhiều khi mình reshape sai nên nó tính sai 
    matching_term = torch.tensor(matching_term_lst)
    # Combine terms
    loss = 0.5 * cross_entropy_term + (1 - matching_term)
    return loss


    
def eval_model(args, model, validation_dataloader):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        
    print("Running Validation...")
    
    model.to(device)
    
    t0 = time.time()
    model.eval()
    total_eval_loss = 0
    total_eval_accuracy = 0
    m = tf.metrics.Accuracy()
    
    for step, batch in enumerate(validation_dataloader):    
        batch = tuple(t.to(device) for t in batch)
        
        b_input_ids, b_input_attention_mask, b_labels = batch    
        print("b_input_ids: ", b_input_ids.shape)
        with torch.no_grad():       
            output = model(b_input_ids, attention_mask=b_input_attention_mask, labels=b_labels) 
            # Assuming b_labels and logits are NumPy arrays
            m.update_state(b_labels.cpu(), torch.argmax(output.logits, dim=-1).cpu())
        
            # b_labels_np = b_labels.cpu().numpy()
            # logits_np = torch.argmax(output.logits, dim=-1).cpu().numpy()
            
        accuracy = m.result().numpy()
        # accuracy = accuracy_score(b_labels_np, logits_np)
        total_eval_loss += output.loss.item()
        total_eval_accuracy += accuracy
        
    avg_eval_loss = total_eval_loss / len(validation_dataloader)
    avg_eval_accuracy = total_eval_accuracy / len(validation_dataloader) 
    return (avg_eval_loss, avg_eval_accuracy)

def train(args, model, optimizer, scheduler, validation_dataloader, train_dataloader):
    # assert args.pregenerated_data.is_file(), \
    #     "--pregenerated_data should point to the folder of files made by pregenerate_training_data.py!"

    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1       
    
    # logging.info("device: {} n_gpu: {}, distributed training: {}, 16-bits training: {}".format(
    #     device, n_gpu, bool(args.local_rank != -1), args.fp16))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(args.gradient_accumulation_steps))

    if args.output_dir.is_dir() and list(args.output_dir.iterdir()):
        logging.warning(f"Output directory ({args.output_dir}) already exists and is not empty!")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    total_time = time.time()
    train_steps = 0
    
    m = tf.metrics.Accuracy()
    m_f1 = tf.metrics.F1Score(num_classes=2, average='micro')
    print('\n========   Evaluate before training   ========')
    
    val_loss, val_accuracy = eval_model(args, model, validation_dataloader)
    tb.add_scalar('validation loss', val_loss)
    tb.add_scalar('validation accucracy', val_accuracy)
    print("Average validiation loss: {:} avg val accuracy {:} : ".format(val_loss, val_accuracy))
    
    # Prepare model
    model = BertForMaskedLM.from_pretrained(args.bert_model)
    model.to(device)
    
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    logging.info("***** Running training *****")
    logging.info(f" Num examples = {args.num_samples}")
   

    loss_dict = defaultdict(list)
    for epoch in range(args.epochs):
        total_train_loss  = 0 
        total_train_accuracy = 0
        print('\n======== Epoch {:} / {:} ========'.format(epoch + 1, args.epochs))
        
        t0 = time.time()
        model.train()
        print('put model in train mode')

        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as pbar:
            for step, batch in enumerate(train_dataloader):
                
                batch = tuple(t.to(device) for t in batch)  
                print("Batch: ", batch)             
                input_ids, input_mask, lm_label_ids = batch     
                model.zero_grad() 
                outputs = model(input_ids=input_ids, attention_mask=input_mask, labels=lm_label_ids)
                # outputs: loss, logits, hidden_states, attentions
                
                loss = outputs.loss
                logits = outputs.logits
                m.update_state(lm_label_ids.cpu(), torch.argmax(logits, dim=-1).cpu())
                m_f1.update_state(lm_label_ids.cpu(), torch.argmax(logits, dim=-1).cpu())
                #b_labels_np = lm_label_ids.cpu().numpy()
                #print("LOGIT TYPE: ", type(logits)) 
            
                #logits_np = torch.argmax(logits, dim=-1).cpu().numpy()
                #print('shape of logits_np: ', logits_np.shape)
                
                # Compute accuracy using scikit-learn's accuracy_score
                accuracy = m.result().numpy()
                #accuracy = accuracy_score(b_labels_np, logits_np)
                
                elapsed = 0
                if step % 50 == 0 and step > 0:
                    elapsed = format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}. loss is {:} accuracy is {:}'.format(step, len(train_dataloader), elapsed, loss.item(), accuracy ))
                
                
                #returns the average loss of batch
                train_steps += len(batch[0])
                
                tb.add_scalar('train loss', outputs.loss, train_steps)
                tb.add_scalar('train accuracy', accuracy, train_steps)
                
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                    
                loss.backward()
                
                total_train_loss += loss.item()
                total_train_accuracy += accuracy
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step() #update the learning rate
                
                # print top 10 masked tokens
                # print(tokenizer.convert_ids_to_tokens(torch.topk(outputs.logits[0, idx, :], 10).indices))
                #print("Input id shape: ", input_ids.shape)  
                #print("Logits shape: ", outputs.logits.shape) # ([32, 85, 28996])  input (N=batch_sz, C=nb_of_class)
                # logits (torch.FloatTensor of shape (batch_size, sequence_length, config.vocab_size)) 
                #print("Target shape: ", lm_label_ids.shape)  # torch.Size([32, 85]) Target:  shape (), (N)
                
                # num_classes = outputs.logits[0].size(1) # Output:  shape (), (N)
                # #print("Num classes ", num_classes) # 28996
                
                
                
                # loss = custom_loss(input_ids=input_ids, logits=outputs.logits.view(-1, num_classes), labels=lm_label_ids.view(-1))
                
                # #loss = outputs[0]
                
                # if args.fp16:
                #     optimizer.backward(loss)
                # else:
                #     loss.backward()
                # total_train_loss += loss.item()
                # nb_tr_examples += input_ids.size(0)
                # nb_tr_steps += 1
                # pbar.update(1)
                # mean_loss = total_train_loss * args.gradient_accumulation_steps / nb_tr_steps
                # pbar.set_postfix_str(f"Loss: {mean_loss:.5f}")
                # if (step + 1) % args.gradient_accumulation_steps == 0:
                #     optimizer.step()
                #     scheduler.step()  # Update learning rate schedule
                #     optimizer.zero_grad()
                #     global_step += 1
                # loss_dict["epoch"].append(epoch)
                # loss_dict["batch_id"].append(step)
                # loss_dict["mlm_loss"].append(loss.item())
            
            # Save a trained model
            if epoch < args.epochs and (n_gpu > 1 and torch.distributed.get_rank() == 0 or n_gpu <= 1):
                logging.info("** ** * Saving fine-tuned model ** ** * ")
                epoch_output_dir = args.output_dir / f"epoch_{epoch}"
                epoch_output_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(epoch_output_dir)
            #   tokenizer.save_pretrained(epoch_output_dir)
                
            avg_train_loss = total_train_loss / len(train_dataloader) 
            avg_train_accuracy = total_train_accuracy / len(train_dataloader) 
            training_time = format_time(time.time() - t0)
            print("  Average training loss: {:} Average training accuracy: {:} Training epcoh took: {:}".format(avg_train_loss,avg_train_accuracy, training_time))
            
            
            val_loss, val_accuracy = eval_model(args, model, validation_dataloader)
            tb.add_scalar('validation loss', val_loss, epoch)
            tb.add_scalar('validation accucracy', val_accuracy, epoch)
            print("Average validiation loss: {:} avg val accuracy {:} : ".format(val_loss, val_accuracy))
    
    # Save a trained model
    if n_gpu > 1 and torch.distributed.get_rank() == 0 or n_gpu <=1:
        logging.info("** ** * Saving fine-tuned model ** ** * ")
        model.save_pretrained(args.output_dir)
    #     tokenizer.save_pretrained(args.output_dir)
    #     df = pd.DataFrame.from_dict(loss_dict)
    #     df.to_csv(args.output_dir/"losses.csv")
def prepare_data(args):
    
    val_dataset = PregeneratedDataset(training_path=args.pregenerated_data, file_name='dev_mlm.json', tokenizer=tokenizer,  reduce_memory=args.reduce_memory)
    
    validation_dataloader = DataLoader(
        val_dataset, 
        sampler=SequentialSampler(val_dataset), 
        batch_size=args.train_batch_size, 
        num_workers=NUM_CPU)
    
    # prepare train dataloader
    epoch_dataset = PregeneratedDataset(training_path=args.pregenerated_data, file_name='train_mlm.json', tokenizer=tokenizer,
                                            reduce_memory=args.reduce_memory)
    if args.local_rank == -1:
        train_sampler = RandomSampler(epoch_dataset)
    else:
        train_sampler = DistributedSampler(epoch_dataset)
    train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=NUM_CPU)
    
    
    test_dataset = PregeneratedDataset(training_path=args.pregenerated_data, file_name='test_mlm.json', tokenizer=tokenizer,  reduce_memory=args.reduce_memory)
    
    test_data_loader = DataLoader(
        test_dataset, 
        sampler=SequentialSampler(test_dataset), 
        batch_size=args.train_batch_size, 
        num_workers=NUM_CPU)
    
    return validation_dataloader, train_dataloader, test_data_loader


def pretrain_on_treatment(args):
    model = BertForMaskedLM.from_pretrained(args.bert_model)
    
    validation_dataloader, train_dataloader, test_dataloader = prepare_data(args)
    print("len dataloader: ", len(validation_dataloader), len(test_dataloader), len(train_dataloader))
 
    # Prepare parameters
    args.train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    num_train_optimization_steps = math.ceil(args.num_samples/args.train_batch_size) // args.gradient_accumulation_steps
    # num_train_optimization_steps = len(train_dataloader) // (args.gradient_accumulation_steps * args.epochs)
    print("Num train optimization steps: ", num_train_optimization_steps)
    
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']  
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
   
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=num_train_optimization_steps)
    
    train(args, model, optimizer, scheduler, validation_dataloader, train_dataloader)
    
    test_loss, test_accuracy = eval_model(args, model, test_dataloader)
    print(f'Test loss: {test_loss} Test accuracy: {test_accuracy}')



def main():
    parser = ArgumentParser()
    parser.add_argument('--pregenerated_data', type=Path, required=False)
    parser.add_argument("--output_dir", type=Path, required=False)
    parser.add_argument("--bert_model", type=str, required=False, default='dmis-lab/biobert-base-cased-v1.2',
                        help="Bert pre-trained model")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")
    parser.add_argument("--num_samples", type=int, required=False, default=39921)
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Number of epochs to train for")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--train_batch_size",
                        default=BATCH_SIZE,
                        type=int,
                        help="Size of each batch.")
    # parser.add_argument('--fp16',
    #                     action='store_true',
    #                     help="Whether to use 16-bit float precision instead of 32-bit")
    # parser.add_argument('--loss_scale',
    #                     type=float, default=0,
    #                     help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
    #                          "0 (default value): dynamic loss scaling.\n"
    #                          "Positive power of 2: static loss scaling value.\n")
    parser.add_argument("--warmup_steps",
                        default=0,
                        type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--adam_epsilon",
                        default=1e-8,
                        type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--learning_rate",
                        default=1e-5,
                        type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--corpus_type", type=str, required=False, default="")
    args = parser.parse_args()
    
    #args.output_dir = Path('/content/drive/MyDrive/Colab Notebooks/mlm_finetune_output')/ 'model'
    args.output_dir = Path('mlm_finetune_output') / "model"
    #args.pregenerated_data = pathlib.Path('/content/drive/MyDrive/Colab Notebooks/mlm_prepare_data')
    args.pregenerated_data = pathlib.Path('mlm_prepared_data')
    
    # data_split('mlm_output', 'mlm_prepared_data', tokenizer)[0]
    pretrain_on_treatment(args)
   

if __name__ == '__main__':
    main()