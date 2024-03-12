import argparse
import random
import time
import os
import numpy as np

import torch

from logger_ import make_logger
from datetime import datetime
from embedding import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader, BatchSampler, RandomSampler, SequentialSampler

from utils.data_utils import METRICS
from transformers import BertForMaskedLM, BertForSequenceClassification, BertTokenizerFast, get_linear_schedule_with_warmup
def make_arguments(parser):
    parser.add_argument('--data_dir', type = str, required=True,
                        help='path to directory where prepared data is present')
    parser.add_argument('--out_dir', type = str, required=True,
                        help = 'path to save the model')
    parser.add_argument('--epochs', type = int, required=True,
                        help = 'number of epochs to train')
    parser.add_argument('--freeze_shared_model', default=False, action='store_true',
                        help = "True to freeze the loaded pre-trained shared model and only finetune task specific headers")
    parser.add_argument('--train_batch_size', type = int, default=32,
                        help='batch size to use for training')
    parser.add_argument('--eval_batch_size', type = int, default = 32,
                        help = "batch size to use during evaluation")
    parser.add_argument('--grad_accumulation_steps', type =int, default = 1,
                        help = "number of steps to accumulate gradients before update")
    parser.add_argument('--num_of_warmup_steps', type=int, default = 0,
                        help = "warm-up value for scheduler")
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                       help = "learning rate for optimizer")
    parser.add_argument('--epsilon', type=float, default=1e-8,
                       help="epsilon value for optimizer")
    parser.add_argument('--grad_clip_value', type = float, default=1.0,
                        help = "gradient clipping value to avoid gradient overflowing" )
    parser.add_argument('--log_file', default='multi_task_logs.log', type = str,
                        help = "name of log file to store")
    parser.add_argument('--log_per_updates', default = 10, type = int,
                        help = "number of steps after which to log loss")
    parser.add_argument('--seed', default=42, type = int,
                        help = "seed to set for modules")
    parser.add_argument('--max_seq_len', default=128, type =int,
                        help = "max seq length used for model at time of data preparation")
    parser.add_argument('--save_per_updates', default = 0, type = int,
                        help = "to keep saving model after this number of updates")
    parser.add_argument('--limit_save', default = 10, type = int,
                        help = "max number recent checkpoints to keep saved")
    parser.add_argument('--load_saved_model', type=str, default=None,
                        help="path to the saved model in case of loading from saved")
    parser.add_argument('--eval_while_train', default = False, action = 'store_true',
                        help = "if evaluation on dev set is required during training.")
    parser.add_argument('--test_while_train', default=False, action = 'store_true',
                        help = "if evaluation on test set is required during training.")
    parser.add_argument('--resume_train', default=False, action = 'store_true',
                        help="Set for resuming training from a saved model")
    
    parser.add_argument('--debug_mode', default = False, action = 'store_true', 
                        help = "record logs for debugging if True")
    parser.add_argument('--silent', default = False, action = 'store_true', 
                        help = "Only write logs to file if True")
    return parser


parser = argparse.ArgumentParser()
parser = make_arguments(parser)
args = parser.parse_args()

#setting logging
now = datetime.now()
logDir = now.strftime("%d_%m-%H_%M")
if not os.path.isdir(logDir):
    os.makedirs(logDir)

logger = make_logger(name = "masked language model", debugMode=args.debug_mode,
                    logFile=os.path.join(logDir, args.log_file), silent=args.silent)
logger.info("logger created.")

device = torch.device('cpu')       

#setting seed
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device('cuda')

assert os.path.isdir(args.data_dir), "data_dir doesn't exists"

if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)
    
def eval_model(model, validation_dataloader):
    print("Running Validation...")
    model.cuda()
    t0 = time.time()
    model.eval()
    total_eval_loss = 0
    total_eval_accuracy = 0
    nb_eval_steps = 0
    # m = tf.metrics.Accuracy()
    # m = Accuracy()
    for step, batch in enumerate(validation_dataloader):    
        batch = tuple(t.to(device) for t in batch)
        b_input_ids, b_token_type_ids, b_input_attention_mask, b_labels = batch    
        with torch.no_grad():       
            (loss, logits) = model(b_input_ids, attention_mask=b_input_attention_mask, token_type_ids=b_token_type_ids, labels=b_labels) 
            
            m.update_state(b_labels.cpu(), torch.argmax(logits, dim=-1).cpu())
        accuracy = m.result().numpy()
        
        total_eval_loss += loss.item()
        total_eval_accuracy += accuracy
    avg_eval_loss = total_eval_loss / len(validation_dataloader)
    avg_eval_accuracy = total_eval_accuracy / len(validation_dataloader) 
    return (avg_eval_loss, avg_eval_accuracy)
    
def train(model, optimizer, scheduler,  train_dataloader, validation_dataloader, epochs = 10, print_every=50):
    model.cuda()
    total_time = time.time() 
    train_steps = 0
    #m = tf.metrics.Accuracy()
    tensorboard = SummaryWriter(log_dir = os.path.join(logDir, 'tb_logs'))
    #m = Accuracy()
    print('\n========   before training   ========')
    val_loss, val_accuracy = eval_model(model, validation_dataloader)
    
    tensorboard.add_scalar('validation loss', val_loss)
    tensorboard.add_scalar('validation accucracy', val_accuracy)
    
    print("Average validiation loss: {:} avg val accuracy {:} : ".format(val_loss, val_accuracy))
    for epoch_i in range(0, epochs):
        total_train_loss = 0
        total_train_accuracy = 0
        print('\n======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        t0 = time.time()
        model.train()
        print('put model in train mode')
        for step, batch in enumerate(train_dataloader):
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_token_type_ids, b_input_attention_mask, b_labels = batch
            model.zero_grad()     
            loss, logits = model(b_input_ids, attention_mask=b_input_attention_mask, token_type_ids=b_token_type_ids, labels=b_labels) 
            
            
            accuracy = METRICS['classification_accuracy'](b_labels, torch.argmax(logits, dim=-1))
                
                
            # m.update_state(b_labels.cpu(), torch.argmax(logits, dim=-1).cpu())
            # accuracy = m.result().numpy()
            # if step % print_every == 0 and step > 0:
            #         elapsed = format_time(time.time() - t0)
            #         print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}. loss is {:} accuracy is {:}'.format(step, len(train_dataloader), elapsed, loss.item(), accuracy ))
        #returns the average loss of batch 
        train_steps += len(batch[0])
        
        tensorboard.add_scalar('train loss', loss, train_steps)
        tensorboard.add_scalar('train accuracy', accuracy, train_steps)
        
        total_train_loss += loss.item()
        total_train_accuracy += accuracy
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step() #update the learning rate
        avg_train_loss = total_train_loss / len(train_dataloader) 
        avg_train_accuracy = total_train_accuracy / len(train_dataloader) 
        training_time = format_time(time.time() - t0)
        print("  Average training loss: {:} Average training accuracy: {:} Training epcoh took: {:}".format(avg_train_loss,avg_train_accuracy, training_time))
        val_loss, val_accuracy = eval_model(model, validation_dataloader)
        tensorboard.add_scalar('validation loss', val_loss, epoch_i)
        tensorboard.add_scalar('validation accucracy', val_accuracy, epoch_i)
        print("Average validiation loss: {:} avg val accuracy {:} : ".format(val_loss, val_accuracy))
     
def read_prepare_data_train(models, epochs=4, lr=1e-5, print_every=100, batch_size=128):
    
    # sentences = df.text.values
    # sentences = [[sent] for sent in sentences]
    # data = encode_sentences(sentences)
    # data_maskedLM = generate_labels_for_maskedLM(data)
    # dataset = prepare_data(model, data_maskedLM)
    # train_dataloader, validation_dataloader, test_dataloader =  split_datatset(dataset, batch_size=batch_size) 
    
    
    # read json file
    train_dataset = read_data('./MLM/mlm_prepared_data/train_mlm.json')
    val_dataset = read_data('./MLM/mlm_prepared_data/dev_mlm.json')
    test_dataset = read_data('./MLM/mlm_prepared_data/test_mlm.json')
    
    train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size)
    
    validation_dataloader = DataLoader(
              val_dataset, # The validation samples.
              sampler = SequentialSampler(val_dataset), # Pull out batches sequentially.
              batch_size = batch_size # Evaluate with this batch size.
          )
    test_dataloader = DataLoader(
              test_dataset, # The validation samples.
              sampler = SequentialSampler(test_dataset), # Pull out batches sequentially.
              batch_size = batch_size # Evaluate with this batch size.
          )
    
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    optimizer = torch.optim.AdamW(models.parameters(), lr = lr,  # args.learning_rate - remember the default is 5e-5
                    eps = 1e-8                      # args.adam_epsilon  - default is 1e-8.
                    )
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = 0, # default value in run_glue.py 
                                                num_training_steps = total_steps)
    train(models, optimizer, scheduler, train_dataloader, validation_dataloader,    epochs = epochs, print_every=print_every)

    test_loss, test_accuracy = eval_model(models, test_dataloader)
    print(f'Test loss: {test_loss} Test accuracy: {test_accuracy}')

models = BertForMaskedLM.from_pretrained('dmis-lab/biobert-base-cased-v1.2')
#model.save_pretrained("/content/drive/My Drive/Colab Notebooks/classification/mlm/")
def main():
    read_prepare_data_train(models, epochs=5, lr=1e-2, print_every=50, batch_size=128)

if __name__ == "__main__":
    main()