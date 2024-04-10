import gc
import os
import pandas as pd
import torch
import time
from tqdm import tqdm

from MLM.model.custom_loss import is_POS_match, CustomLoss
from MLM.model.custom_dataset import CustomDataset

def eval_model(args, logger, model, epoch, loss_fn=CustomLoss, validation_dataloader=CustomDataset, wrt_path=None):
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
       
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        
    model.to(device)
    model.eval()
    
    t0 = time.time()
    total_loss = 0
    batch_num=0
    all_pred_pos_tag_is = []
    all_origin_pos_tag_id = []
    all_pred_id = []
    all_origin_id = []
    
    for batch in tqdm(validation_dataloader, total=len(validation_dataloader), desc = 'Eval'):
      
      batch = tuple(t.to(device) for t in batch)
      b_input_ids, b_input_attention_mask, b_token_type_id, b_labels = batch

      # step 1. compute the output    
      with torch.no_grad():   
          output = model(b_input_ids, 
                         attention_mask=b_input_attention_mask, 
                         token_type_ids = b_token_type_id, 
                         labels=b_labels) 
      
      # get pos tag prediction
      _, b_pred_id, b_pred_pos_tag_id, b_origin_pos_tag_id = is_POS_match(b_input_ids, output.logits, b_labels)   
      all_pred_pos_tag_is.extend(b_pred_pos_tag_id)
      all_origin_pos_tag_id.extend(b_origin_pos_tag_id)
      all_pred_id.extend(b_pred_id)
      all_origin_id.extend(b_input_ids)
      
      
      # step 2. compute the loss
      loss_batch = loss_fn(output.logits, b_input_ids, b_labels)
      if n_gpu > 1:
          loss_batch = loss_batch.mean() # mean() to average on multi-gpu.
      
      total_loss += loss_batch
      batch_num += 1
      
     
    assert len(all_pred_pos_tag_is) == len(all_origin_pos_tag_id) == len(all_pred_id) == len(all_origin_id), logger.debug("lengths are not equal")
    
    avg_eval_loss = total_loss / batch_num
    val_time = time.time() - t0
    logger.info("  Average validate loss: {:} Training epoch took: {:} secs ".format(avg_eval_loss, val_time))
    
    del total_loss
    gc.collect()
    if args.pred_dir is not None and wrt_path is not None:
        df = pd.DataFrame({"prediction_pos_tag_id" : [t.cpu().numpy() for t in all_pred_pos_tag_is], "label_pos_tag_id" : [t.cpu().numpy() for t in all_origin_pos_tag_id], 
                            "prediction_id" : [t.cpu().numpy() for t in all_pred_id], "origin_id" : [t.cpu().numpy() for t in all_origin_id]})
        
        savePath = os.path.join(args.pred_dir, "pred_mlm_{}_{}.tsv".format(wrt_path, epoch))
        df.to_csv(savePath, sep = "\t", index = False)
    
    del all_pred_pos_tag_is, all_origin_pos_tag_id, all_pred_id, all_origin_id
    return avg_eval_loss