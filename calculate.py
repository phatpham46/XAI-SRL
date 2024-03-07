from sklearn.metrics import f1_score
import ast, os

def read_pred_file(filePath):
    data = []
    predictions = []
    labels = []
    # filePath = "./output2/"+ filePath
    with open(filePath, "r") as f:
        for line in f:
            # delete the first line
            if "uid" in line:
                continue
            uid, prediction, label = line.strip().split("\t")
            labels.append(label)
            predictions.append(prediction)
            data.append((prediction.strip("[]"), label.strip("[]")))
    return labels, predictions


def processing_data(labels, predictions):
    lineTokens = [ast.literal_eval(label) for label in labels]
    lineLabels = [ast.literal_eval(prediction) for prediction in predictions]

    trueLabels = [] # labels
    predictLabels = [] # predictions

    for lineTok, lineLab in zip(lineTokens, lineLabels):
        if lineLab in ['[CLS]','[SEP]', 'X']: # replace non-text tokens with O. These will not be evaluated.
            predictLabels.append('O')
            trueLabels.append('O')
            continue
        if(lineLab == "B-V"):
            predictLabels.append("V")
        else:
            predictLabels.append(lineLab)
            trueLabels.append(lineTok) 
    return trueLabels, predictLabels


def main():
    dirPath = "./output_SRL_finetuned"
    files = []
    for path in os.listdir(dirPath):
        if os.path.isfile(os.path.join(dirPath, path)):
            files.append(path)
            
       
    y_pred = []
    y_true = []
    for file in files:
        if "test_" not in file:
            continue
        labels, predictions = read_pred_file(os.path.join(dirPath, file))
        trueLabels, predLabels = processing_data(labels, predictions)
        
        trueLabels = [item for sublist in trueLabels for item in sublist]
        predLabels = [item for sublist in predLabels for item in sublist]
        
        y_true.append(trueLabels) 
        y_pred.append(predLabels)
        
    y_true = [item for sublist in y_true for item in sublist]
    y_pred = [item for sublist in y_pred for item in sublist]
    result_f1 = f1_score(y_true, y_pred, average="micro")    
    print("F1 score: ", result_f1) 
   


if __name__ == '__main__':
    
    main()
    