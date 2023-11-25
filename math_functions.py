import numpy as np
from constants import *

def binaryCrossEntropy(output, y_train):
    n = y_train.shape[0]
    
    result = -1 / n
    sum = np.sum(y_train * np.log(output)) + ((1 - y_train) * np.log(1 - output))
    return result * sum

def crossEntropy(output, y_train):
    return -np.sum(y_train * np.log(output))

def entropy(output, y_train):
    return -np.sum(y_train * np.log(y_train))

def softmax(z):
    assert len(z.shape) == 2
    s = np.max(z, axis=1)
    s = s[:, np.newaxis] # necessary step to do broadcasting
    e_x = np.exp(z - s)
    div = np.sum(e_x, axis=1)
    div = div[:, np.newaxis] # dito
    return e_x / div

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def softmax(Z):
    expZ = np.exp(Z.T)
    return expZ / np.sum(expZ)

def derivative_cost(output, waited):
    return 2 * (output - waited)

def derivative(Z, fn):
    if fn == SIGMOID:
        fn = sigmoid
    elif fn == SOFTMAX:
        fn = softmax
    return fn(Z) * (1 - fn(Z))

def cost_function(output, waited):
    m = waited.shape[0]
    cost = -(1/m)*np.sum(waited * np.log(output))
    return cost