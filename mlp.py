import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from Layer import Layer
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

    dZ = binaryCrossEntropy(output, waited_output)
    # dZ = (output - waited_output)
    dZ = np.expand_dims(dZ, axis=1)
    # print("DZ:", dZ)
    input = np.expand_dims(input, axis=1)
    dW = (1/m) * dZ.dot(input.T)
    dB = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    return (dZ, dW, dB) 

def check_data(data):
    data = data.drop(0, axis=1)
    # create y train (waited output of our neural network)
    y_check = data[1].copy()
    y_check = y_check[round(data.shape[0] * percentage_from_data):data.shape[0]]
    y_check = y_check.replace('M', 1)
    y_check = y_check.replace('B', 0)
    y_check = np.array(y_check)
    data = data.drop(1, axis=1)
    # initialize x values (input of our neural network)
    x_check = pd.DataFrame(data[round(data.shape[0] * percentage_from_data):data.shape[0]].values)
    x_check = np.array(x_check)
    x_check = normalize_data(x_check)
    return y_check, x_check

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
    for i in range(500):
        sum_last_dW, sum_last_dB, sum_h1_dW, sum_h1_dB, sum_dW, sum_dB = 0,0,0,0,0,0
        for j in range(x_train.shape[0]):
            x_instance = np.tile(x_train[j], (10, 1))

            outputInputLayer = inputLayer.computeLayer(x_train[j])
            outputHiddenLayer = hiddenLayer1.computeLayer(outputInputLayer)
            final = outputLayer.computeLayer(outputHiddenLayer)

            # print(cost_function(final, y_train[0]))
            last_dZ, last_dW, last_dB = backpropagation_last_layer(final, y_train[j], outputHiddenLayer)
            h1_dZ, h1_dW, h1_dB = backpropagation_layer(last_dZ, outputLayer.weights, outputHiddenLayer, outputInputLayer)
            dZ,dW, dB = backpropagation_layer(h1_dZ, hiddenLayer1.weights, outputInputLayer, x_instance)
            sum_last_dW += last_dW
            sum_last_dB += last_dB
            sum_h1_dW += h1_dW
            sum_h1_dB += h1_dB
            sum_dW += dW
            sum_dB += dB 

        outputLayer.update_params(sum_last_dB / x_train.shape[0], sum_last_dW  / x_train.shape[0])
        hiddenLayer1.update_params(sum_h1_dB / x_train.shape[0], sum_h1_dW / x_train.shape[0])
        inputLayer.update_params(sum_dB / x_train.shape[0], sum_dW / x_train.shape[0])

    y_check, x_check = check_data(data)
    for i in range(x_check.shape[0]):
        x_instance = np.tile(x_check[i], (10, 1))

        outputInputLayer = inputLayer.computeLayer(x_check[i])
        outputHiddenLayer = hiddenLayer1.computeLayer(outputInputLayer)
        final = outputLayer.computeLayer(outputHiddenLayer)
        print("waited value: ", y_check[i], " output value: ", final)

if __name__ == "__main__":
    main()
    
#plot data

# binary cross entropy

    
    
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
