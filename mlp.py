import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from Layer import Layer
import math
from constants import *

epochs = 90
layer = 3 
node_per_layer = 30
learning_rate = 0.15
batch_size = 100

def binaryCrossEntropy(output, y_train):
    n = y_train.shape[0]
    
    sum = 0
    for i in range(n):
        pred = output[0, :][0]
        sum += y_train[i] * np.log(pred) + ((1 - y_train[i]) * np.log(1 - pred))
    return sum  / n * -1
    # return result * sum

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def softmax(Z):
    print("shape:", Z.shape)
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
    return (x_train)

def derivative_cost(output, waited):
    return 2 * (output - waited)

def derivative(Z, fn):
    if fn == SIGMOID:
        fn = sigmoid
    elif fn == SOFTMAX:
        fn = softmax
    return fn(Z) * (1 - fn(Z))

# def optimal_weights(output, waited, sub_output, fn):
#     if fn == SIGMOID:
#         activation = sigmoid
#     elif fn == SOFTMAX:
#         activation = softmax
    
#     cost_res_weights = sub_output * derivative(output, fn) * derivative_cost(output, waited)
#     return cost_res_weights

def init_data(data):
    data = data.drop(0, axis=1)
    
    # create y train (waited output of our neural network)
    y_train = data[1].copy()
    y_train = y_train[:batch_size]
    y_train = y_train.replace('M', 1)
    y_train = y_train.replace('B', 0)
    y_train = np.array(y_train)
    data = data.drop(1, axis=1)
    # initialize x values (input of our neural network)
    x_train = pd.DataFrame(data[:batch_size].values)
    x_valid = pd.DataFrame(data[batch_size:data.shape[0]].values)
    x_train = np.array(x_train)
    x_train = normalize_data(x_train)
    return x_train, y_train

def prediction(data):
    data = data.drop(0, axis=1)
    y_check = data[1].copy()
    y_check = y_check[batch_size:data.shape[0]]
    y_check = y_check.replace('M', 1)
    y_check = y_check.replace('B', 0)
    y_check = np.array(y_check)
    data = data.drop(1, axis=1)
    x_check = pd.DataFrame(data[batch_size:data.shape[0]].values)
    x_check = np.array(x_check)
    for i in range(x_check.shape[0]):
        outputHiddenLayer = hiddenLayer.computeLayer(x_check[i])
        final = outputLayer.computeLayer(outputHiddenLayer)
        print("Waited: ", y_check[i], " Get: ",  final)

def main():
    if len(sys.argv) != 2:
        print("Wrong number of args")
        exit (1)

    data = pd.read_csv(sys.argv[1], header=None)
    x_train, y_train = init_data(data)
    y_train_one_hot = one_hot(y_train)

    inputLayer = Layer(node_per_layer, x_train.shape[0], sigmoid)
    hiddenLayer = Layer(node_per_layer, node_per_layer,sigmoid)
    outputLayer = Layer(2, node_per_layer, softmax)
    
    # x_instance = x_train[0]
    final = np.array([0, 0])
    for j in range(1000):
    # for i in range(x_train.shape[0]):
        output_inputLayer = inputLayer.computeLayer(x_train)
        output_hiddenLayer = hiddenLayer.computeLayer(output_inputLayer)
        final = outputLayer.computeLayer(output_hiddenLayer)

        if j % 100 == 0:
            entropy = binaryCrossEntropy(final, y_train)
            print("Entropy:", entropy)
        
        deltaFinal = outputLayer.deltaOutputLayer(y_train_one_hot, final)
        deltaHidden = hiddenLayer.deltaHiddenLayer(deltaFinal, output_hiddenLayer, outputLayer.weights)

        inputLayer.update_weights(output_inputLayer, inputLayer.deltaHiddenLayer(deltaHidden, output_inputLayer, hiddenLayer.weights))
        hiddenLayer.update_weights(output_hiddenLayer, deltaHidden)
        outputLayer.update_weights(final, deltaFinal)
    print("Final result:", final)

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
