import numpy as np

from parsing import init_data

def accuracy(y_accuracy, final):
    #final[:, 0] same as final.T[0]
    diff = y_accuracy - final[:, 0]
    sum = np.sum(np.abs(diff))

    precision = sum / final.shape[0]
    return precision

def binaryCrossEntropy(y_pred, y_train):
    n = y_train.shape[0]
    
    sum = 0
    for i in range(n):
        pred = y_pred[i][0]
        sum += y_train[i] * np.log(pred) + ((1 - y_train[i]) * np.log(1 - pred))
    return sum  / n * -1