import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from Layer import Layer
import math

epochs = 90
layer = 3 
node_per_layer = 24
learning_rate = 0.15
percentage_from_data = 0.85

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

def cost(output, y_train):
    cost = (output - y_train) ** 2
    return cost

def print_shape(data, x_train, x_valid):
    print("initial shape:", data.shape)
    print("x_train shape:", x_train.shape)
    print("x_valid shape:", x_valid.shape)
    
def print_shape(x_train):
    min_val = np.min(x_train)
    max_val = np.max(x_train)
    x_train = (x_train - min_val) / (max_val - min_val)

def main():
    data = pd.read_csv(sys.argv[1], header=None)
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

    inputLayer = Layer(10, x_train.shape[0],sigmoid)
    hiddenLayer1 = Layer(10, 10,sigmoid)
    hiddenLayer2 = Layer(10, 10,sigmoid)
    hiddenLayer3 = Layer(10, 10,sigmoid)
    # outputLayer = Layer(10, softmax, 10)
    outputLayer = Layer(1, 1,sigmoid)

    final = []
    # for x_instance in x_train:
    delta = inputLayer.computeLayer(x_train)
    delta = hiddenLayer1.computeLayer(delta)
    delta = hiddenLayer2.computeLayer(delta)
    delta = hiddenLayer3.computeLayer(delta)
    final.append(outputLayer.computeLayer(delta))

        # test = test[np.newaxis, :] # dito
        # print(softmax(test))
        # print(softmax(test).sum())
    final = np.array(final)
    dZ = final - y_train
    dW =  1 / y_train.shape[0] * dZ * final.T
    # print(dZ)
    # for x_instance in x_train:
        # dZ = 

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
