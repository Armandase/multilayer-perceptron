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

# def softmax(z):
#     assert len(z.shape) == 2
#     s = np.max(z, axis=1)
#     s = s[:, np.newaxis] # necessary step to do broadcasting
#     e_x = np.exp(z - s)
#     div = np.sum(e_x, axis=1)
#     div = div[:, np.newaxis] # dito
#     return e_x / div

def sigmoid(x):
    return (1 / (1 + np.exp(-x)))

def cost(output, y_train):
    cost = (output - y_train) ** 2
    return cost

def main():
    data = pd.read_csv(sys.argv[1], header=None)
    data = data.drop(0, axis=1)
    y_train = data[1].copy()
    y_train = y_train.replace('M', 1)
    y_train = y_train.replace('B', 0)
    data = data.drop(1, axis=1)
    x_train = pd.DataFrame(data[0:round(data.shape[0] * 0.85)].values)
    x_valid = pd.DataFrame(data[x_train.shape[0]:data.shape[0]].values)
    print("initial shape:", data.shape)
    print("x_train shape:", x_train.shape)
    print("x_valid shape:", x_valid.shape)

    x_train = np.array(x_train)
    # min_val = np.min(x_train)
    # max_val = np.max(x_train)
    # x_train = (x_train - min_val) / (max_val - min_val)
    inputLayer = Layer(10, sigmoid, 10)
    hiddenLayer1 = Layer(10, sigmoid, 10)
    hiddenLayer2 = Layer(10, sigmoid, 10)
    hiddenLayer3 = Layer(10, sigmoid, 10)
    # outputLayer = Layer(10, softmax, 10)
    outputLayer = Layer(1, sigmoid, 10)

    # for i in range(epochs):
    for x_instance in x_train:
        delta = inputLayer.computeLayer(x_instance)
        delta = hiddenLayer1.computeLayer(delta)
        delta = hiddenLayer2.computeLayer(delta)
        delta = hiddenLayer3.computeLayer(delta)
        final = outputLayer.computeLayer(delta)
        print(final)
        print(binaryCrossEntropy(final, y_train))
        outputLayer.cost(final, y_train)

if __name__ == "__main__":
    main()