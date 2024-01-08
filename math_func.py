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
    
    sum = np.sum(y_train * np.log(y_pred[:, 0]) + ((1 - y_train) * np.log(1 - y_pred[:, 0])))
    return sum  / n * -1

def meanSquareError(y_pred, y_train):
    n = y_train.shape[0]
    
    diff = y_train - y_pred[:, 0]
    sum = np.sum(diff ** 2)
    return sum  / n