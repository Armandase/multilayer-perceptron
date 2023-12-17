import sys
import pandas as pd
import numpy as np
from sklearn import preprocessing
from Layer import Layer
from parsing import *
import math
from constants import *
import random

epochs = 300
node_per_layer = 20
learning_rate = 0.0001
batch_size = 100
nb_feature = 30

def binaryCrossEntropy(output, y_train):
    n = y_train.shape[0]
    
    sum = 0
    for i in range(n):
        pred = output[i][0]
        sum += y_train[i] * np.log(pred) + ((1 - y_train[i]) * np.log(1 - pred))
    return sum  / n * -1

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def softmax(Z):
    exp_sum = np.sum(np.exp(Z), axis=1)
    for i in range(exp_sum.shape[0]):
        if np.isnan(exp_sum[i]):
            print(Z)
            exit()
    y_pred = np.empty([Z.shape[0], 2])
    for i in range(Z.shape[0]):
        y_pred[i] = np.exp(Z[i]) / exp_sum[i]
    return y_pred

def main():
    if len(sys.argv) != 2:
        print("Wrong number of args")
        exit (1)

    random.seed(10)
    np.random.seed(10)
    try:
        data = pd.read_csv(sys.argv[1], header=None)
    except:
        print("Parsing error.") 
        exit(1)
    x_train, y_train = init_data(data, batch_size)

    inputLayer = Layer(node_per_layer, nb_feature, sigmoid)
    hiddenLayer = Layer(node_per_layer, node_per_layer,sigmoid)
    outputLayer = Layer(2, node_per_layer, softmax)

    for j in range(epochs):
        

        inputLayer.computeLayer(x_train)
        hiddenLayer.computeLayer(inputLayer.output)
        final = outputLayer.computeLayer(hiddenLayer.output)

        if j % 100 == epochs / (len(str(epochs)) * 10):
            entropy = binaryCrossEntropy(final, y_train)
            print("Entropy:", entropy)
        
        deltaFinal = outputLayer.deltaOutputLayer(y_train)
        deltaHidden = hiddenLayer.deltaHiddenLayer(deltaFinal)
        deltaInput = inputLayer.deltaHiddenLayer(deltaHidden)

        inputLayer.update_weights(deltaInput)
        hiddenLayer.update_weights(deltaHidden)
        outputLayer.update_weights(deltaFinal)

    sum = 0
    for i in range(final.shape[0]):
        print("Waited: ", y_train[i], " Get: ",  final[i])
        if y_train[i] == 1:
            sum += y_train[i] - final[i][0]
        else:
            sum += final[i][0]
    print("Accurancy: ", sum / final.shape[0])


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
