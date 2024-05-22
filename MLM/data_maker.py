
import numpy as np
import sys
sys.path.append('../')
from data_preparation import * 
from mlm_utils.pertured_dataset import PerturedDataset
import torch.nn as nn
import math
import os
import torch

class DataMaker():
    def __init__(self, data_file, out_dir, eval_batch_size=32, max_seq_len=85, seed=42):
        self.data_file = data_file
        self.out_dir = out_dir
        self.eval_batch_size = eval_batch_size
        self.max_seq_len = max_seq_len
        self.seed = seed
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        assert os.path.exists(self.data_file), "prediction tsv file not present at {}".format(self.data_file)
        
        self.dataset = PerturedDataset(self.data_file, self.device)
        self.dataloader = self.dataset.generate_batches(self.dataset, self.eval_batch_size)
       
    def get_predictions(self, model, is_masked=False):
        model.network.eval()
        
        allPreds = []
        allScores = []
        allLogitsSoftmax = []
        allLogitsRaw = []
        allLabels = []
        allOriginUIDs = []
        for batch in tqdm(self.dataloader, total = len(self.dataloader)):
            batch = tuple(t.to(self.device) if isinstance(t, torch.Tensor) else t for t in batch)

            if is_masked: 
                origin_uid, token_id, mask, type_id, pos_tag_id = batch
                # create a dummy label tensor
                label = torch.zeros(token_id.size(0), token_id.size(1), dtype=torch.long).to(self.device)
            else: 
                origin_uid, label, token_id, type_id, mask = batch 
            
            with torch.no_grad():
                _, logits = model.network(token_id, type_id, mask, 0, 'conllsrl')

               
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
            break

        labMapRevN = {0: 'B-A1',
                        1: 'I-A1',
                        2: 'O',
                        3: 'B-V',
                        4: 'B-A0',
                        5: 'I-A0',
                        6: 'B-A4',
                        7: 'I-A4',
                        8: 'I-A3',
                        9: 'B-A2',
                        10: 'I-A2',
                        11: 'B-A3',
                        12: '[CLS]',
                        13: '[SEP]',
                        14: 'X'}

        print("allPreds: ", len(allPreds), len(allPreds[0]))
        
        for j, (p, l) in enumerate(zip(allPreds, allLabels)):
            allLabels[j] = l[:len(p)]
            allPreds[j] = [labMapRevN[int(ele)] for ele in p]
            allLabels[j] = [labMapRevN[int(ele)] for ele in allLabels[j]]
        #allPreds[i] = [ [ labMapRev[int(p)] for p in pp ] for pp in allPreds[i] ]
        #allLabels[i] = [ [labMapRev[int(l)] for l in ll] for ll in allLabels[i] ]

        newPreds = []
        newLabels = []
        newScores = []
        newLogitsSoftmax = []
        for m, samp in enumerate(allLabels):
            Preds = []
            Labels = []
            Scores = []
            LogitsSm = []
            for n, ele in enumerate(samp):
                #print(ele)
                if ele != '[CLS]' and ele != '[SEP]' and ele != 'X':
                    #print('inside')
                    Preds.append(allPreds[m][n])
                    Labels.append(ele)
                    Scores.append(allScores[m][n])
                    LogitsSm.append(allLogitsSoftmax[m][n])
                    #del allLabels[i][m][n]
                    #del allPreds[i][m][n]
            newPreds.append(Preds)
            newLabels.append(Labels)
            newScores.append(Scores)
            newLogitsSoftmax.append(LogitsSm)
        allLabels = newLabels
        allPreds = newPreds
        allScores = newScores    
        allLogitsSoftmax = newLogitsSoftmax        
                
        # flatten allPreds, allScores
        allOriginUIDs = [item for sublist in allOriginUIDs for item in sublist]
        allPreds = [item for sublist in allPreds for item in sublist]
        allScores = [item for sublist in allScores for item in sublist]
        allLogitsSoftmax = [item for sublist in allLogitsSoftmax for item in sublist]
        allLogitsRaw = [item for sublist in allLogitsRaw for item in sublist]
        allLabels = [item for sublist in allLabels for item in sublist]
        return allOriginUIDs, allPreds, allScores, allLogitsSoftmax, allLogitsRaw, allLabels    