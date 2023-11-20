import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from Layer import Layer
import math
from constants import *

epochs = 90
layer = 3 
node_per_layer = 24
learning_rate = 0.15
percentage_from_data = 0.85

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
    expZ = np.exp(Z)
    return expZ / np.sum(expZ)

def one_hot(y_train):
    one_hot = np.zeros((y_train.size, y_train.max() + 1))
    n_values = np.max(y_train) + 1
    one_hot = np.eye(n_values)[y_train]
    return one_hot

def print_shape(data, x_train, x_valid):
    print("initial shape:", data.shape)
    print("x_train shape:", x_train.shape)
    print("x_valid shape:", x_valid.shape)
    
def normalize_data(x_train):
    min_val = np.min(x_train)
    max_val = np.max(x_train)
    x_train = (x_train - min_val) / (max_val - min_val)

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
    print(cost_res_weights)
    return cost_res_weights

def init_data(data):
    data = data.drop(0, axis=1)
    
    # create y train (waited output of our neural network)
    y_train = data[1].copy()
    y_train = y_train[:round(data.shape[0] * percentage_from_data)]
    y_train = y_train.replace('M', 1)
    y_train = y_train.replace('B', 0)
    y_train = np.array(y_train)
    data = data.drop(1, axis=1)
    # initialize x values (input of our neural network)
    x_train = pd.DataFrame(data[0:round(data.shape[0] * percentage_from_data)].values)
    x_valid = pd.DataFrame(data[x_train.shape[0]:data.shape[0]].values)
    x_train = np.array(x_train)
    # x_train = normalize_data(x_train)
    return x_train, y_train

def main():
    if len(sys.argv) != 2:
        print("Wrong number of args")
        exit (1)

    data = pd.read_csv(sys.argv[1], header=None)
    x_train, y_train = init_data(data)
    y_train_one_hot = one_hot(y_train)
    
    inputLayer = Layer(10, x_train.shape[1], sigmoid)
    hiddenLayer1 = Layer(10, 10,sigmoid)
    # outputLayer = Layer(10, softmax, 10)
    outputLayer = Layer(2, 10, softmax)
    
    # for x_instance in x_train:
    x_instance = x_train[0]
    x_instance = np.tile(x_instance, (10, 1))
    delta = inputLayer.computeLayer(x_instance.T)
    delta = hiddenLayer1.computeLayer(delta)
    final = outputLayer.computeLayer(delta)

    mPercentage = np.sum(final[0])
    bPercentage = np.sum(final[1])
    
    m_cost = (mPercentage - y_train_one_hot[0][0]) ** 2
    b_cost = (bPercentage - y_train_one_hot[0][1]) ** 2

    tmp = np.array([mPercentage, bPercentage])
    tmp2 = np.array([y_train_one_hot[0][0], y_train_one_hot[0][1]])

    updated_weights = optimal_weights(tmp, tmp2, delta[0][0], SOFTMAX)
    # dZ = final[0] - y_train[0]
    # dW =  1 / y_train[0] * dZ * final[0]

if __name__ == "__main__":
    main()
    
# back prop:
# dZ[n] : error in last layer
# A :  matrice des output du réseau (softmax)
# Y : matrice des outputs attendus
# [n] : couche du réseau
# m : number of value in dZ (equal to A)
# g' (Z) : derivative of activation function 

# dZ[n] = A[n] - Y (just for last layer)
# dZ[0..n-1] = (W[n] Transpose) * dZ[n]
# dW[n] = 1 / m  *  dZ[n] * (A[n-1] Tranpose)
# db[n] = 1 / m * sum (dZ[n])

# to update weigts and biais:
# W[n] = W[n] - learning rate * dW[n]
# b[n] = b[n] - learning rate * db[n]
