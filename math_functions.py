import numpy as np
from constants import *


def binaryCrossEntropy(output, y_train):
    result = - 1/y_train.shape[0]
    sum = 0
    mean = 0
    for element in output:
        mean += element
    mean /= len(output)
    for i in range(y_train.shape[0]):
        sum += (y_train[i] * math.log(mean)) + ((1 - y_train[i]) * math.log(1 - mean))
    return result * sum

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

def optimal_weights(output, waited, sub_output, fn):
    if fn == SIGMOID:
        activation = sigmoid
    elif fn == SOFTMAX:
        activation = softmax
    
    cost_res_weights = sub_output * derivative(output, fn) * derivative_cost(output, waited)
    return cost_res_weights

def cost_function(output, waited):
    m = waited.shape[0]
    cost = -(1/m)*np.sum(waited * np.log(output))
    return cost