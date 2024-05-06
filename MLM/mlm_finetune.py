import math
import time
import torch
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# sys.path.append('../')
# sys.path.append('/content/SRLPredictionEasel')
sys.path.append('/kaggle/working/SRLPredictionEasel')


from logger_ import make_logger
# sys.path.insert(1, '../')
# sys.path.insert(1, '/content/SRLPredictionEasel')
sys.path.insert(1, '/kaggle/working/SRLPredictionEasel')
from tqdm import tqdm
from pathlib import Path
from datetime import datetime
from argparse import ArgumentParser
from model.custom_loss import CustomLoss
from model.custom_dataset import CustomDataset
from model.eval import eval_model
from torch.utils.tensorboard import SummaryWriter
from transformers import get_linear_schedule_with_warmup 
from mlm_utils.transform_func import check_data_dir
from mlm_utils.model_utils import BATCH_SIZE, EPOCHS, BIOBERT_MODEL, BERT_PRETRAIN_MODEL


def make_argument(parser):
    parser.add_argument('--data_dir', type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--pred_dir", type=Path, required=True)
    parser.add_argument("--log_dir", type=Path, required=False, 
                        default=Path("logs_mlm"))
  
    parser.add_argument("--model_file", type=Path, required=False)
    parser.add_argument("--bert_model", type=str, required=False, 
                        default=BERT_PRETRAIN_MODEL,
                        help="Bert pre-trained model")
    parser.add_argument("--do_lower_case", action="store_true")
    parser.add_argument("--reduce_memory", action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")
    parser.add_argument("--num_samples", type=int, required=False, 
                        default=0, help="Number of samples in the dataset")
    parser.add_argument("--epochs", type=int, default=EPOCHS,
                        help="Number of epochs to train for")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--train_batch_size",
                        default=BATCH_SIZE,
                        type=int,
                        help="Size of each batch.")
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
    parser.add_argument('--debug_mode', default = False, action = 'store_true', 
                        help = "record logs for debugging if True")
    parser.add_argument('--log_file', default='mlm_finetune_logs.log', type = str, 
                        help = "name of log file to store")
    parser.add_argument('--silent', default = False, action = 'store_true', 
                        help = "Only write logs to file if True")
    parser.add_argument("--corpus_type", type=str, required=False, default="")
    parser.add_argument("--load_save_model", default=False, action='store_true', 
                        help = "Load saved model to continue training")
    
    return parser

parser = ArgumentParser()
parser = make_argument(parser)
args = parser.parse_args()

# setting tensorboard
now = datetime.now()
logDir = args.log_dir / now.strftime("%d_%m-%H_%M")
check_data_dir(logDir, auto_create=True)
tb = SummaryWriter(logDir / "tb_logs")


# setting logging
logger = make_logger(name = "mlm_finetune", debugMode=args.debug_mode,
                    logFile=os.path.join(logDir, args.log_file), silent=args.silent)
logger.info("logger created.")


def save_model(model, optimizer, scheduler, globalStep, min_val_loss, savePath) :
    modelStateDict = {k : v.cpu() for k,v in model.state_dict().items()}
    toSave = {'model_state_dict' :modelStateDict,
            'optimizer_state' : optimizer.state_dict(),
            'scheduler_state' : scheduler.state_dict(),
            'global_step' : globalStep,
            'min_val_loss' : min_val_loss}
    
    torch.save(toSave, savePath)
    logger.info('model saved in {} global step at {}'.format(globalStep, savePath))

def load_model(loadPath, model, device, optimizer, scheduler):
    loadedDict = torch.load(loadPath, map_location=device)
 
    loadedDict['model_state_dict'] = {k.lstrip('module.'):v for k, v in loadedDict['model_state_dict'].items()}
   
    model.load_state_dict(loadedDict['model_state_dict'])
    optimizer.load_state_dict(loadedDict['optimizer_state'])
    scheduler.load_state_dict(loadedDict['scheduler_state'])    
    
    
    min_val_loss = loadedDict['min_val_loss']   
    return min_val_loss
            
def train(args, model, optimizer, scheduler, min_val_loss, loss_fn:CustomLoss, val_dataset:CustomDataset, train_dataset:CustomDataset, test_dataset:CustomDataset):
   
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1       
    
    logger.info("device: {} n_gpu: {}".format(device, n_gpu))

    if args.output_dir.is_dir() and list(args.output_dir.iterdir()):
        logger.warning(f"Output directory ({args.output_dir}) already exists and is not empty!")
        
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    if args.pred_dir.is_dir() and list(args.pred_dir.iterdir()):
        logger.warning(f"Prediction directory ({args.pred_dir}) already exists and is not empty!")
        
    args.pred_dir.mkdir(parents=True, exist_ok=True)
    
    
    global_step = 0 # across all batches
    
    model.to(device)
    
    if n_gpu > 1:
        model = torch.nn.DataParallel(model)

    args.num_samples = len(train_dataset)
    logger.info("***** Running training *****")
    logger.info(f" Num examples = {args.num_samples}")
   
    num_train_steps = math.ceil(args.num_samples / args.train_batch_size) * args.epochs 
    logger.info("Num train optimization steps: {}".format(num_train_steps))
    
    for epoch in range(args.epochs):
        
        logger.info('\n================= EPOCH {:} ================='.format(epoch))

        model.train()
        print('put model in train mode')

        logger.info("Create data for training...")
        train_dataloader = train_dataset.generate_batches(
            local_rank= args.local_rank, 
            dataset= train_dataset,           
            batch_size=args.train_batch_size)
       
        val_dataloader = val_dataset.generate_batches(
            local_rank= args.local_rank, 
            dataset= val_dataset,
            batch_size=args.train_batch_size)
        
        test_dataloader = test_dataset.generate_batches(
            local_rank= args.local_rank, 
            dataset= test_dataset,
            batch_size= args.train_batch_size)
        
        total_train_loss  = 0 
      
        batch_num = 0
        t0 = time.time()
        with tqdm(total=len(train_dataloader), desc=f"Epoch {epoch}") as progress:
            for batch in train_dataloader:
                batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)

                # batch = tuple(t.to(device) for t in batch)  
                b_mask_input_id, b_attention_mask, b_token_type_id, b_label_id = batch 
                
                # step 1: zero the gradient  
                optimizer.zero_grad()
                
                # step 2: compute the output
                outputs = model(b_mask_input_id, 
                                attention_mask=b_attention_mask,
                                token_type_ids = b_token_type_id, 
                                labels=b_label_id) 
                
                # step 3: compute the loss
                # loss = custom_loss(args,b_logit_id=outputs.logits, 
                #                    b_input_id=b_mask_input_id, 
                #                    b_label_id=b_label_id)
                loss = loss_fn(outputs.logits, b_mask_input_id, b_label_id)
              
                if n_gpu > 1:
                    loss = loss.mean() # mean() to average on multi-gpu.
                    
                    
                total_train_loss += loss.item()
                
                # step 4: use loss to produce gradients 
                loss.backward()
                
                
                # step 5: use optimizer to take gradient step
                optimizer.step()
                scheduler.step() 
                optimizer.zero_grad()
            
                global_step += 1
                batch_num += 1
                progress.update(1)

        training_time = time.time() - t0
        
        # Average loss per epoch
        avg_train_loss = total_train_loss / batch_num
        
        # validation
        logger.info("\nRunning Evaluation on validation... at epoch {}".format(epoch))
        avg_val_loss = eval_model(args, logger, model, epoch, loss_fn, val_dataloader, wrt_path=None)        
            
        logger.info("  Average training loss: {:} Training epoch took: {:}".format(avg_train_loss, training_time))
        
        # visualize loss
        logger.info("Visualizing loss of training and valuating set.. at epoch {}".format(epoch))
        tb.add_scalar('train/val mlm loss', avg_train_loss, epoch)
        tb.add_scalar('train/val mlm loss', avg_val_loss, epoch)
        
        
        # Testing model
        logger.info("\nRunning Evaluation on test... at epoch {}".format(epoch))
        test_loss = eval_model(args, logger, model, epoch, loss_fn, test_dataloader, wrt_path = "test_predictions_mlm")
        
        # Save model after each epoch
        if avg_val_loss < min_val_loss :
            min_val_loss = avg_val_loss
            
        logger.info("  Min validation loss: {:} at epoch {:}".format(min_val_loss, epoch))    
        if  epoch < args.epochs and (n_gpu > 1 and torch.distributed.get_rank() == 0 or n_gpu <= 1):
            logger.info("** ** * Saving fine-tuned model ** ** * ")
            epoch_output_dir = args.output_dir / f"mlm_epoch_{epoch}.pt"
            
            save_model(model, optimizer, scheduler, global_step, min_val_loss, epoch_output_dir)
            
            logger.info('model saved in {} global step at {}'.format(global_step, epoch_output_dir))
  
def main():
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Prepare model
    model = BIOBERT_MODEL
    model.to(device)
    
    # Prepare data
    train_dataset = CustomDataset(
        data_path=args.data_dir, 
        file_name='train_mlm.json')
    
    validation_dataset = CustomDataset(
        data_path=args.data_dir,
        file_name='dev_mlm.json')
    
    test_dataset = CustomDataset(
        data_path=args.data_dir,
        file_name='test_mlm.json')
    
    logger.info("Length of train dataset: {}".format(len(train_dataset)))
    logger.info("Length of validation dataset: {}".format(len(validation_dataset)))
    logger.info("Length of test dataset: {}".format(len(test_dataset)))
    
    
    # Prepare parameters
    num_train_steps = math.ceil(args.num_samples / args.train_batch_size) * args.epochs 
    total_steps = int(num_train_steps / args.epochs)
    
    
    # Prepare optimizer
    param_optimizer = list(BIOBERT_MODEL.named_parameters())
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
    
    # Prepare loss 
    loss_fn = CustomLoss()
    logger.info("\nARGS: {}".format(args))
    
   
    # assign min threshold with max value
    if args.load_save_model and args.model_file.is_file():
        min_val_loss = load_model(args.model_file, model, device, optimizer, scheduler)
        
        logger.info("Model loaded successfully")
        train(args, model, optimizer, scheduler, min_val_loss, loss_fn, validation_dataset, train_dataset, test_dataset)
    else:    
        # Train model
        min_val_loss = float('inf')
        train(args, model, optimizer, scheduler, min_val_loss, loss_fn, validation_dataset, train_dataset, test_dataset)

def test_multiprocess(x):
    return x*x

if __name__ == '__main__':
    # python mlm_finetune.py --data_dir mlm_prepared_data_3/ --output_dir mlm_finetune_output_3 --pred_dir 
    # python mlm_finetune.py --data_dir mlm_prepared_data_3/ --output_dir mlm_finetune_output_3 --pred_dir mlm_predition --load_save_model --model_file ././mlm_finetune_output/model/mlm_epoch_4.pt

    main()
    
    