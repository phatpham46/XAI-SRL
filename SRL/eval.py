import os
import math
import logging
import pandas as pd
from tqdm import tqdm
from utils.data_utils import METRICS, TaskType

logger = logging.getLogger("srl_task")
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
    allIds = [[] for _ in range(numTasks)]
    allLogits = [[] for _ in range(numTasks)]
    for batchMetaData, batchData in tqdm(dataLoader, total=numStep, desc = 'Eval'):
        batchMetaData, batchData = batchSampler.patch_data(batchMetaData,batchData, gpu = gpu)
        prediction, logitSm = model.predict_step(batchMetaData, batchData)

        logger.debug("predictions in eval: {}".format(prediction))       
        batchTaskId = int(batchMetaData['task_id'])
        
        orgLabels = batchMetaData['label']
        allLabels[batchTaskId].extend(orgLabels)
        allLabels_str[batchTaskId].extend(orgLabels)
        logger.debug("batch task id in eval: {}".format(batchTaskId))
        allPreds[batchTaskId].extend(prediction)
        allIds[batchTaskId].extend(batchMetaData['uids'])
        allLogits[batchTaskId].extend(logitSm)
        
        
    for i in range(len(allPreds)):
        if allPreds[i] == []:
            continue
        taskName = taskParams.taskIdNameMap[i]
        taskType = taskParams.taskTypeMap[taskName]
        labMap = taskParams.labelMap[taskName]

        if taskType == TaskType.SRL:
        # SRL requires label clipping. We''ve already clipped our predictions
        # using attn Masks, so we will clip labels to predictions len
        # Also we need to remove the extra tokens from predictions based on labels
            labMapRevN = {v:k for k,v in labMap.items()}

            for j, (p, l) in enumerate(zip(allPreds[i], allLabels[i])):
                allLabels_str[i][j] = l[:len(p)]
                allPreds[i][j] = [labMapRevN[int(ele)] for ele in p]
                allLabels[i][j] = [labMapRevN[int(ele)] for ele in allLabels_str[i][j]]
                allLogits[i][j] = allLogits[i][j][:len(p)]
           
            newPreds = []
            newLabels = []
            newLogits = []
            newLabels_str = []
            for m, samp in enumerate(allLabels[i]):
                Preds = []
                Labels = []
                Logits = []
                labStr = []
                for n, ele in enumerate(samp):
                    if ele != '[CLS]' and ele != '[SEP]' and ele != 'X':
                        Preds.append(allPreds[i][m][n])
                        Labels.append(ele)
                        labStr.append(allLabels_str[i][m][n])
                        Logits.append(allLogits[i][m][n])
                newPreds.append(Preds)
                newLabels.append(Labels)
                newLogits.append(Logits)
                newLabels_str.append(labStr)
            allLabels[i] = newLabels
            allPreds[i] = newPreds
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
                    logger.info("{} : {}".format(m, metricVal))
                    brier_score = METRICS['brier_score'](allLabels_str[i], allLogits[i])
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
            
    if returnPred:
        return allIds, allPreds