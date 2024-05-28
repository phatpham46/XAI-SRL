import logging
import pandas as pd
from utils.data_utils import METRICS, TaskType
import math
import os
from tqdm import tqdm
# logger = logging.getLogger("multi_task")
logger = logging.getLogger("evaluate_brier_score")
def evaluate(dataSet, batchSampler, dataLoader, taskParams,
            model, gpu, evalBatchSize, needMetrics, hasTrueLabels,
            wrtDir=None, wrtPredPath = None, returnPred=False):
    '''
    Function to make predictions on the given data. The provided data can be multiple tasks or single task
    It will seprate out the predictions based on task id for metrics evaluation
    '''
    numTasks = len(taskParams.taskIdNameMap)
    numStep = math.ceil(len(dataLoader)/evalBatchSize)
    allPreds = [[] for _ in range(numTasks)]
    allLabels = [[] for _ in range(numTasks)]
    allLabels_str = [[] for _ in range(numTasks)]
    allScores = [[] for _ in range(numTasks)]
    allIds = [[] for _ in range(numTasks)]
    allLogits = [[] for _ in range(numTasks)]
    for batchMetaData, batchData in tqdm(dataLoader, total=numStep, desc = 'Eval'):
        batchMetaData, batchData = batchSampler.patch_data(batchMetaData,batchData, gpu = gpu)
        prediction, scores, logitSm = model.predict_step(batchMetaData, batchData)

        logger.debug("predictions in eval: {}".format(prediction))       
        batchTaskId = int(batchMetaData['task_id'])
        
        orgLabels = batchMetaData['label']
        allLabels[batchTaskId].extend(orgLabels)
        allLabels_str[batchTaskId].extend(orgLabels)
        logger.debug("batch task id in eval: {}".format(batchTaskId))
        allPreds[batchTaskId].extend(prediction)
        allScores[batchTaskId].extend(scores)
        allIds[batchTaskId].extend(batchMetaData['uids'])
        allLogits[batchTaskId].extend(logitSm)
        
        
    for i in range(len(allPreds)):
        if allPreds[i] == []:
            continue
        taskName = taskParams.taskIdNameMap[i]
        taskType = taskParams.taskTypeMap[taskName]
        labMap = taskParams.labelMap[taskName]


        if taskType == TaskType.NER:
        # NER requires label clipping. We''ve already clipped our predictions
        #using attn Masks, so we will clip labels to predictions len
        # Also we need to remove the extra tokens from predictions based on labels
        #print(labMap)
            labMapRevN = {v:k for k,v in labMap.items()}

            for j, (p, l) in enumerate(zip(allPreds[i], allLabels[i])):
                allLabels_str[i][j] = l[:len(p)]
                allPreds[i][j] = [labMapRevN[int(ele)] for ele in p]
                allLabels[i][j] = [labMapRevN[int(ele)] for ele in allLabels_str[i][j]]
                allLogits[i][j] = allLogits[i][j][:len(p)]
            #allPreds[i] = [ [ labMapRev[int(p)] for p in pp ] for pp in allPreds[i] ]
            #allLabels[i] = [ [labMapRev[int(l)] for l in ll] for ll in allLabels[i] ]

            newPreds = []
            newLabels = []
            newScores = []
            newLogits = []
            newLabels_str = []
            for m, samp in enumerate(allLabels[i]):
                Preds = []
                Labels = []
                Scores = []
                Logits = []
                labStr = []
                for n, ele in enumerate(samp):
                    #print(ele)
                    if ele != '[CLS]' and ele != '[SEP]' and ele != 'X':
                        #print('inside')
                        Preds.append(allPreds[i][m][n])
                        Labels.append(ele)
                        Scores.append(allScores[i][m][n])
                        labStr.append(allLabels_str[i][m][n])
                        Logits.append(allLogits[i][m][n])
                        #del allLabels[i][m][n]
                        #del allPreds[i][m][n]
                newPreds.append(Preds)
                newLabels.append(Labels)
                newScores.append(Scores)
                newLogits.append(Logits)
                newLabels_str.append(labStr)
            allLabels[i] = newLabels
            allPreds[i] = newPreds
            allScores[i] = newScores
            allLogits[i] = newLogits
            allLabels_str[i] = newLabels_str
    if needMetrics:
        # fetch metrics from task id
        for i in range(len(allPreds)):
            if allPreds[i] == []:
                continue
            taskName = taskParams.taskIdNameMap[i]
            metrics = taskParams.metricsMap[taskName]
            if metrics is None:
                logger.info("No metrics are provided in task params (file)")
                continue
            if taskName == 'conllsrl' :
                logger.info("********** {} Evaluation************\n".format(taskName))
                for m in metrics:
                    metricVal = METRICS[m](allLabels[i], allPreds[i])
                   
                    brier_score = METRICS['brier_score'](allLabels_str[i], allLogits[i])
                    logger.info("{} : {}".format(m, metricVal))
                    logger.info("Brier Score : {}".format(brier_score))

    if wrtPredPath is not None and wrtDir is not None:
        for i in range(len(allPreds)):
            if allPreds[i] == []:
                continue
            taskName = taskParams.taskIdNameMap[i]
            if hasTrueLabels:
                df = pd.DataFrame({"uid" : allIds[i], "prediction" : allPreds[i], "label" : allLabels[i]})
            else:
                df = pd.DataFrame({"uid" : allIds[i], "prediction" : allPreds[i]})

            savePath = os.path.join(wrtDir, "{}_{}".format(taskName, wrtPredPath))
            df.to_csv(savePath, sep = "\t", index = False)
            #logger.info("Predictions File saved at {}".format(savePath))

    if returnPred:
        return allIds, allPreds, allScores