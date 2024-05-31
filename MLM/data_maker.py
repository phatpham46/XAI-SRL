
import numpy as np 
import pandas as pd
from data_preparation import * 
from mlm_utils.pertured_dataset import PerturbedDataset
import torch.nn as nn
import os
import torch
from mlm_utils.metric_func import brier_score_multi_class

class DataMaker():
    def __init__(self, data_file, eval_batch_size=32, max_seq_len=85, seed=42):
        self.data_file = data_file
        
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        assert os.path.exists(self.data_file), "prediction tsv file not present at {}".format(self.data_file)
        
        self.dataset = PerturbedDataset(self.data_file, self.device)
        self.dataloader = self.dataset.generate_batches(self.dataset, self.eval_batch_size)
       
    def get_predictions(self, model, is_mask_token, del_mask_token):
        model.network.eval()
        
        allPreds = []
        allScores = []
        allLogitsSoftmax = []
        allLogitsRaw = []
        allLabels = []
        allOriginUIDs = []
        for batch in tqdm(self.dataloader, total = len(self.dataloader)):
            batch = tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch)

            origin_uid, token_id, type_id, mask, label, pos_tag_id, masked_id = batch
            
            with torch.no_grad():
                if is_mask_token:
                    _, logits = model.network(masked_id, type_id, mask, 0, 'conllsrl')
                
                elif del_mask_token:
                    
                    # delete mask token (103) in masked_id and padding token 0 at the end
                    masked_id = masked_id.cpu().numpy().tolist()
                    filtered_ids = [[item for item in sublist if item != 103] for sublist in masked_id]

                    padded_ids = pad_sequences(filtered_ids, maxlen=self.max_seq_len, padding='post', truncating='post', value=0)
                   
                    token_id = torch.tensor(padded_ids, dtype=torch.long).to(self.device)
                    _, logits = model.network(token_id, type_id, mask, 0, 'conllsrl')
                    
                else: _, logits = model.network(token_id, type_id, mask, 0, 'conllsrl')

               
                outLogitsSoftmax = nn.functional.softmax(logits, dim = 2).data.cpu().numpy()
                
                outLogitsSigmoid = nn.functional.sigmoid(logits).data.cpu().numpy()
                
                predicted_sm = np.argmax(outLogitsSoftmax, axis = 2)
                

                # here in score, we only want to give out the score of the class of tag, which is maximum
                predScore = np.max(outLogitsSigmoid, axis = 2).tolist() 
                
                predicted_sm = predicted_sm.tolist()
                
                # get the attention masks, we need to discard the predictions made for extra padding
                
                predictedTags = []
                predScoreTags = []
                
                if mask is not None:
                    #shape of attention Masks (batchSize, maxSeqLen)
                    actualLengths = mask.cpu().numpy().sum(axis = 1).tolist()
                
                    for i, (pred, sc) in enumerate(zip(predicted_sm, predScore)):
                        predictedTags.append( pred[:actualLengths[i]] )
                        predScoreTags.append( sc[:actualLengths[i]])
        
                else:
                    predictedTags = predicted_sm
                    predScoreTags = predScore
                
                allPreds.append(predictedTags)  
                allScores.append(predScoreTags)  
                allLogitsSoftmax.append(outLogitsSoftmax)
                allLabels.append(label.tolist())
                allLogitsRaw.append(logits.data.cpu().numpy())
                allOriginUIDs.append(origin_uid)
            
        allOriginUIDs = [item for sublist in allOriginUIDs for item in sublist]
        allPreds = [item for sublist in allPreds for item in sublist]
        allScores = [item for sublist in allScores for item in sublist]
        allLogitsSoftmax = [item for sublist in allLogitsSoftmax for item in sublist]
        allLogitsRaw = [item for sublist in allLogitsRaw for item in sublist]
        allLabels = [item for sublist in allLabels for item in sublist]

        return allOriginUIDs, allPreds, allScores, allLogitsSoftmax, allLogitsRaw, allLabels


    def evaluate(self, model, labMapRevN, wrtPredPath=None, wrtDir=None, returnPreds=True, hasTrueLabels=True, needMetrics=True, is_mask_token=False, del_mask_token=False):
        allOriginUIDs, allPreds, allScores, allLogitsSoftmax, allLogitsRaw, allLabels = self.get_predictions(model, is_mask_token, del_mask_token)
        
        for j, (p, l) in enumerate(zip(allPreds, allLabels)):
            allLabels[j] = l[:len(p)]
            allPreds[j] = [labMapRevN[int(ele)] for ele in p]
            allLabels[j] = [labMapRevN[int(ele)] for ele in allLabels[j]]
        
        newPreds = []
        newLabels = []
        newScores = []
        newLogitsSoftmax = []
        
        labelMap = {v:k for k,v in labMapRevN.items()}
        for m, samp in enumerate(allLabels):
            Preds = []
            Labels = []
            Scores = []
            LogitsSm = []
            for n, ele in enumerate(samp):
                #print(ele)
                if ele != '[CLS]' and ele != '[SEP]' and ele != 'X':
                  
                    Preds.append(allPreds[m][n])
                    Labels.append(ele)
                    Scores.append(allScores[m][n])
                    LogitsSm.append(allLogitsSoftmax[m][n])
                    
            newPreds.append(Preds)
            newLabels.append(Labels)
            newScores.append(Scores)
            newLogitsSoftmax.append(LogitsSm)
        
        allLabels = newLabels
        allPreds = newPreds
        allScores = newScores    
        allLogitsSoftmax = newLogitsSoftmax        
        
        if needMetrics:
        
            print("**********Evaluation************\n")
            
            brier_score_batch = list(map(brier_score_multi_class, allLabels, allLogitsSoftmax))
            brier_score_batch = np.mean(brier_score_batch)
            print("Brier Score: ", brier_score_batch)
        
        # flatten allPreds, allScores
        if wrtPredPath is not None and wrtDir is not None:
            for i in range(len(allPreds)):
                if allPreds[i] == []:
                    continue
                if hasTrueLabels:
                    df = pd.DataFrame({"uid" : allOriginUIDs[i], "prediction" : allPreds[i], "label" : allLabels[i]})
                    savePath = os.path.join(wrtDir, "origin_{}".format(wrtPredPath))
                else:
                    df = pd.DataFrame({"uid" : allOriginUIDs[i], "prediction" : allPreds[i]})
                    savePath = os.path.join(wrtDir, "masked_{}".format(wrtPredPath))

            df.to_csv(savePath, sep = "\t", index = False)
        
        if returnPreds:
            return allOriginUIDs, allPreds, allScores, allLogitsSoftmax, allLogitsRaw, allLabels  