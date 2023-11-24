import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from Layer import Layer
import math
from math_functions import *

epochs = 90
layer = 3 
node_per_layer = 10
learning_rate = 0.15
percentage_from_data = 0.85



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
    return x_train


def init_data(data):
    data = data.drop(0, axis=1)
    # create y train (waited output of our neural network)
    y_train = data[1].copy()
    y_train = y_train[:round(data.shape[0] * percentage_from_data)]
    y_train = y_train.replace('M', 1)
    y_train = y_train.replace('B', 0)
    y_train = np.array(y_train)
    y_train_one_hot = one_hot(y_train)
    data = data.drop(1, axis=1)
    # initialize x values (input of our neural network)
    x_train = pd.DataFrame(data[0:round(data.shape[0] * percentage_from_data)].values)
    x_train = np.array(x_train)
    x_train = normalize_data(x_train)
    return y_train, x_train

def backpropagation_layer(above_deriv_Z, above_weights, output, input):
    m = input.shape[0]
    
    dZ = (1/m) * above_weights.T.dot(above_deriv_Z) * derivative(output, SIGMOID)
    # input = np.expand_dims(input, axis=1)
    dW = (1/m) * dZ.dot(input)
    dB = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    return (dZ, dW, dB) 

def backpropagation_last_layer(output, waited_output, input):
    m = input.shape[0]
    
    dZ = (output - waited_output)
    dZ = np.expand_dims(dZ, axis=1)
    input = np.expand_dims(input, axis=1)
    dW = (1/m) * dZ.dot(input.T)
    dB = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    return (dZ, dW, dB) 

def main():
    if len(sys.argv) != 2:
        print("Wrong number of args")
        exit (1)

    data = pd.read_csv(sys.argv[1], header=None)
    y_train, x_train = init_data(data)
    y_train = one_hot(y_train)
    inputLayer = Layer(node_per_layer, x_train[0].shape[0], sigmoid)
    hiddenLayer1 = Layer(node_per_layer, node_per_layer,sigmoid)
    outputLayer = Layer(2, node_per_layer, softmax)
    
    # for x_instance in x_train:
    # x_instance = x_train[0]
    x_instance = np.tile(x_train[0], (10, 1))
    for i in range(100):
        outputInputLayer = inputLayer.computeLayer(x_train[0])
        outputHiddenLayer = hiddenLayer1.computeLayer(outputInputLayer)
        final = outputLayer.computeLayer(outputHiddenLayer)
        print(final)

        # print(cost_function(final, y_train[0]))
        last_dZ, last_dW, last_dB = backpropagation_last_layer(final, y_train[0], outputHiddenLayer)
        h1_dZ, h1_dW, h1_dB = backpropagation_layer(last_dZ, outputLayer.weights, outputHiddenLayer, outputInputLayer)
        dZ,dW, dB = backpropagation_layer(h1_dZ, hiddenLayer1.weights, outputInputLayer, x_instance)
        outputLayer.update_params(last_dB, last_dW)
        hiddenLayer1.update_params(h1_dB, h1_dW)
        inputLayer.update_params(dB, dW)

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
