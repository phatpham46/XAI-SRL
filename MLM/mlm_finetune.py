import math
import time
import torch
import sys
import logging
import pathlib
import tensorflow as tf
import torch.nn.functional as F
import matplotlib.pyplot as plt


from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from argparse import ArgumentParser
from babel.dates import format_time
from mlm_utils.custom_dataset import CustomDataset
# sys.path.insert(1, '/content/SRLPredictionEasel')
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.distributed import DistributedSampler 
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from transformers import AdamW, get_linear_schedule_with_warmup, BertForMaskedLM 
from mlm_utils.preprocess_functions import get_pos_tag_word, get_key, get_heuristic_word, get_pos_tag_id
from mlm_utils.model_utils import BATCH_SIZE, EPOCHS, BIOBERT_MODEL, BERT_PRETRAIN_MODEL, TOKENIZER, NUM_CPU, MAX_SEQ_LEN
from prepared_for_mlm import get_word_list, data_split, get_tokens_for_words, encode_text, decode_token


sys.path.insert(1, '../')
tb = SummaryWriter()
# tb = SummaryWriter("/content/SRLPredictionEasel/MLM/logs")




def is_POS_match(args, b_input_id, b_logit_id, b_label_id):
    '''
    Function to check if the POS tag of the masked token in the logits is the same as the POS tag of the masked token in the original text.
    Note: This function assumes that the logits are of shape # ([85, 28996]) 
    lm_label_ids: shape (batch_size, sequence_length)
    '''
    '''cho 1 batch'''
    
    b_matching_term = []
    for idx_sample in range(b_input_id.shape[0]):
        
        input_id = b_input_id[idx_sample]
        logit_id = b_logit_id[idx_sample]
        label_id = b_label_id[idx_sample]
        
        pred_id = input_id.clone() 
        origin_input_id = input_id.clone()
        
        # Find the index of the masked token from lm_label_ids
        mask_index = torch.where(label_id != -100)[0]
        masked_idx_input = torch.where(input_id == TOKENIZER.mask_token_id)[0]
        
        # make sure masked_idx_input and mask_index are the same using asser
        assert torch.equal(mask_index, masked_idx_input), "Masked index and label index are not the same."
        origin_input_id[masked_idx_input] = label_id[mask_index] 
        
        
        # "================= ORIGINAL ============= ")
        # get pos tag of origin text
        origin_text = decode_token(origin_input_id, skip_special_tokens=True) 
        
        # Get masked word in the sentence
        masked_word = decode_token(origin_input_id[mask_index])
        
        pos_tag_origin = get_pos_tag_word(masked_word, origin_text)
        
        word_list = get_word_list(origin_text)
        word_dict = {i: torch.tensor(TOKENIZER.encode_plus(i,add_special_tokens = False)['input_ids'], dtype=torch.int64) for i in word_list}

        pos_tag_id_origin = get_pos_tag_id(args, word_dict, pos_tag_origin, label_id)
        
        # "-============== PREDICTION =================="
        pred = [torch.argmax(logit_id[i]).item() for i in mask_index]

        # Replace the index of the masked token with the list of predicted tokens
        for i in mask_index:
            pred_id[i] = pred[i - mask_index[0]]
         
         
        pos_tag_dict_pred = get_pos_tag_word(decode_token(pred), decode_token(pred_id, skip_special_tokens=True) )
        
        pred_sentence = decode_token(pred_id, skip_special_tokens=True)
       
        pred_word_list = get_word_list(pred_sentence)
        
        word_dict_pred = {i: torch.tensor(TOKENIZER.encode_plus(i, add_special_tokens = False)['input_ids'], dtype=torch.int64) for i in pred_word_list}
       
        # get pos tag for all tokens of each word
        pos_tag_id_pred = get_pos_tag_id(args, word_dict_pred, pos_tag_dict_pred, pred_id)
       
        matching_term_tensor = torch.zeros_like(pos_tag_id_pred)
        
        matching_term_tensor[mask_index] = torch.where(pos_tag_id_pred[mask_index] == pos_tag_id_origin[mask_index], 
                                       torch.tensor(1), 
                                       torch.tensor(0))

        b_matching_term.append(matching_term_tensor)
        
    return b_matching_term


def custom_loss(args, b_logit_id, b_input_id, b_label_id):
   
    # Cross-entropy term
    b_cross_entropy_term = F.cross_entropy((1-b_logit_id).view(-1, TOKENIZER.vocab_size), b_label_id.view(-1), reduction='none')
   
    # Custom matching term
    b_matching_term = torch.stack(is_POS_match(args, b_input_id, b_logit_id, b_label_id)).view(-1)

    # Combine terms
    b_loss = 0.5 * ((b_cross_entropy_term) + (1 - b_matching_term))
    return b_loss.mean()


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
    total_eval_loss = 0
    total_eval_accuracy = 0
    model.eval()
    
    m = tf.metrics.Accuracy()
    
    for batch_index, batch in enumerate(validation_dataloader):   
         
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_input_attention_mask, b_labels = batch  
        
        # step 1. compute the output    
        with torch.no_grad():   
            output = model(b_input_ids, attention_mask=b_input_attention_mask, labels=b_labels) 
            
            # Assuming b_labels and logits are NumPy arrays
            m.update_state(b_labels.cpu(), torch.argmax(output.logits, dim=-1).cpu())
        
            # b_labels_np = b_labels.cpu().numpy()
            # logits_np = torch.argmax(output.logits, dim=-1).cpu().numpy()
            
        # step 2. compute the loss
        loss = output.loss
        loss_batch = loss.item()
        total_eval_loss += (loss_batch - total_eval_loss) / (batch_index + 1)
        
        # step 3: compute the accuracy
        accuracy = m.result().numpy()
        total_eval_accuracy += (accuracy - total_eval_accuracy) / (batch_index + 1)
        
    # avg_eval_loss = total_eval_loss / len(validation_dataloader)
    # avg_eval_accuracy = total_eval_accuracy / len(validation_dataloader) 
    return (total_eval_loss, total_eval_accuracy)

def visualize_acc(loss_dict):
    '''
    Function to visualize loss and accuracy for each epoch
    Input:
        dict: 
    '''
    
    # Extract data
    epochs = loss_dict["epoch"]
    loss = loss_dict["mlm_loss"]
    accuracy = loss_dict["mlm_acc"]

    # Plotting loss
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, loss, marker='o', label='MLM Loss')
    plt.title('MLM Loss per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plotting accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(epochs, accuracy, marker='o', color='r', label='MLM Accuracy')
    plt.title('MLM Accuracy per Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()
        
def train(args, model, optimizer, scheduler, val_dataset, train_dataset):
   
   
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
    m_f1 = tf.metrics.F1Score()
    
    
    # print('\n========   Evaluate before training   ========')
    
    # val_loss, val_accuracy = eval_model(args, model, validation_dataloader)
    # tb.add_scalar('validation loss', val_loss)
    # tb.add_scalar('validation accucracy', val_accuracy)
    # print("Average validiation loss: {:} avg val accuracy {:} : ".format(val_loss, val_accuracy))
    
    # Prepare model
    model = BIOBERT_MODEL
    model.to(device)
    
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    global_step = 0
    logging.info("***** Running training *****")
    logging.info(f" Num examples = {args.num_samples}")
   

    loss_dict = defaultdict(list)
    for epoch in range(args.epochs):
        total_train_loss  = 0 # running loss
        total_train_accuracy = 0  # running accuracy
        print('\n======== EPOCH {:} / {:} ========'.format(epoch + 1, args.epochs))
       
        num_train_steps = math.ceil(args.num_samples / args.train_batch_size) * args.epochs // args.gradient_accumulation_steps
    
        total_steps = int(num_train_steps * args.gradient_accumulation_steps / args.epochs)
        t0 = time.time()
        model.train()
        print('put model in train mode')

        train_dataloader = generate_batches(
            local_rank= args.local_rank, 
            dataset=train_dataset, 
            batch_size=args.train_batch_size, 
            device=device)
        
        with tqdm(total=total_steps,position=epoch, desc=f"Epoch {epoch}") as progress:
            for batch_index, batch in enumerate(train_dataloader):
                
                batch = tuple(t.to(device) for t in batch)  
                b_mask_input_id, b_attention_mask,b_token_type_id, b_label_id = batch 
                
                # step 1: zero the gradient  
                optimizer.zero_grad()
                
                # step 2: compute the output
                outputs = model(b_mask_input_id, 
                                attention_mask=b_attention_mask,
                                token_type_ids = b_token_type_id, 
                                labels=b_label_id) 
                
                # step 3: compute the loss
                # loss = outputs.loss
                loss = custom_loss(args,b_logit_id=outputs.logits, 
                                   b_input_id=b_mask_input_id, 
                                   b_label_id=b_label_id)
                print("Loss:", loss)
                # visualize loss
                tb.add_scalar('train loss', loss, global_step)
                global_step += 1


                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                
                # step 4: use loss to produce gradients 
                loss.backward()
                
                # step 5: use optimizer to take gradient step
                if (batch_index + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    scheduler.step() 
                    optimizer.zero_grad()
                
                # -----------------------------------------------------------
                # Compute the accuracy
                logits = outputs.logits
                
                m.update_state(b_label_id.cpu(), torch.argmax(logits, dim=-1).cpu())
                m_f1.update_state(b_label_id.cpu(), torch.argmax(logits, dim=-1).cpu())
                
                accuracy_batch = m.result().numpy()
                
                loss_batch = loss.item()
                total_train_loss += (loss_batch - total_train_loss) / (batch_index + 1)
               
                elapsed = 0
                if batch_index % 50 == 0 and batch_index > 0:
                    elapsed = format_time(time.time() - t0)
                    print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}. loss is {:} accuracy is {:}'.format(batch_index, len(train_dataloader), elapsed, loss.item(), accuracy_batch ))
                
                
                # returns the average loss of batch
                train_steps += len(batch[0])
                
                tb.add_scalar('train loss', outputs.loss, train_steps)
                tb.add_scalar('train accuracy', accuracy_batch, train_steps)
                
                total_train_accuracy += (accuracy_batch - total_train_accuracy) / (batch_index + 1)
                
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                loss_dict["epoch"].append(epoch)
                loss_dict["batch_id"].append(batch_index)
                loss_dict["mlm_loss"].append(total_train_loss)
                loss_dict["mlm_acc"].append(total_train_accuracy)
                progress.update(1)
               
            
            # Save a trained model
            if epoch < args.epochs and (n_gpu > 1 and torch.distributed.get_rank() == 0 or n_gpu <= 1):
                logging.info("** ** * Saving fine-tuned model ** ** * ")
                epoch_output_dir = args.output_dir / f"epoch_{epoch}"
                epoch_output_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(epoch_output_dir)
                
                
            # avg_train_loss = total_train_loss / len(train_dataloader)
            # avg_train_accuracy = total_train_accuracy / len(train_dataloader)
            training_time = format_time(time.time() - t0)
            print("  Average training loss: {:} Average training accuracy: {:} Training epcoh took: {:}".format(total_train_loss, total_train_accuracy, training_time))
            
            
            # val_loss, val_accuracy = eval_model(args, model, validation_dataloader)
            # tb.add_scalar('validation loss', val_loss, epoch)
            # tb.add_scalar('validation accucracy', val_accuracy, epoch)
            # print("Average validiation loss: {:} avg val accuracy {:} : ".format(val_loss, val_accuracy))
    
    # Visualize loss and accuracy
    visualize_acc(loss_dict)
    
    # Save a trained model
    if n_gpu > 1 and torch.distributed.get_rank() == 0 or n_gpu <=1:
        logging.info("** ** * Saving fine-tuned model ** ** * ")
        model.save_pretrained(args.output_dir)
    
        # tokenizer.save_pretrained(args.output_dir)
        # df = pd.DataFrame.from_dict(loss_dict)
        # df.to_csv(args.output_dir/"losses.csv")

def get_sampler(local_rank, dataset):
    if local_rank == -1:
        return RandomSampler(dataset)
    else:
        return SequentialSampler(dataset)
    

def generate_batches(local_rank, dataset, batch_size,
    drop_last=True, device="cpu"):
    """
    A generator function which wraps the PyTorch DataLoader. It will
    ensure each tensor is on the write device location.
    """
    dataloader = DataLoader(
        dataset=dataset, 
        sampler=get_sampler(local_rank, dataset),
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=NUM_CPU)  

    return dataloader

def pretrain_on_treatment(args, model):
   
    # Prepare data
    
    train_dataset = CustomDataset(
        data_path=args.data_dir, 
        file_name='train_mlm.json')
    
    validation_dataset = CustomDataset(
        data_path=args.data_dir,
        file_name='dev_mlm.json')
    
    
    # Prepare parameters
    num_train_steps = math.ceil(args.num_samples / args.train_batch_size) * args.epochs // args.gradient_accumulation_steps
    total_steps = int(num_train_steps * args.gradient_accumulation_steps / args.epochs)
    print("Num train optimization steps: ", total_steps)
    
    
    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']  
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
   
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=args.warmup_steps,
                                                num_training_steps=total_steps)
    
    # Train model
    train(args, model, optimizer, scheduler, validation_dataset, train_dataset)
    
    
    # # Evaluate model
    # test_loss, test_accuracy = eval_model(args, model, test_dataloader)
    # print(f'Test loss: {test_loss} Test accuracy: {test_accuracy}')



def main():
    parser = ArgumentParser()
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--bert_model", type=str, required=False, default=BERT_PRETRAIN_MODEL,
                        help="Bert pre-trained model")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")
    parser.add_argument("--num_samples", type=int, required=False, default=51823)
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
    
    
    # args.output_dir = Path('/content/drive/MyDrive/ColabNotebooks/mlm_finetune_output')/ 'model'
   
    # args.output_dir = Path('mlm_finetune_output') / "model"
    # args.data_dir = pathlib.Path('/content/drive/MyDrive/ColabNotebooks/mlm_prepare_data')
    # args.data_dir = pathlib.Path('mlm_prepared_data_3')
    
    pretrain_on_treatment(args, BIOBERT_MODEL)
   

if __name__ == '__main__':
    main()