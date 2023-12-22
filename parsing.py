import numpy as np
import pandas as pd
import random

def one_hot(y_train):
    one_hot = np.zeros((y_train.size, y_train.max() + 1))
    n_values = np.max(y_train) + 1
    one_hot = np.eye(n_values)[y_train]

    one_hot = np.where(one_hot == 1, 0, 1)
    return one_hot

def normalize_data(x_train):
    min_list = np.min(x_train, axis=0)
    max_list = np.max(x_train, axis=0)
    x_train = (x_train - min_list) / (max_list - min_list)
    return (x_train)

def init_data(data_x, data_y, batch_size):
    # create y train (waited output of our neural network)
    indexes = np.random.randint(0, data_x.shape[0], batch_size)
    y_train = data_y[indexes]
    x_train = data_x.iloc[indexes].values

    x_train = normalize_data(np.array(x_train))
    y_train = np.array(y_train)
    return x_train, y_train

def prediction(data, inputLayer, hiddenLayer, outputLayer):
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
        output_inputLayer = inputLayer.computeLayer(x_check[i])
        output_hiddenLayer = hiddenLayer.computeLayer(output_inputLayer)
        final = outputLayer.computeLayer(output_hiddenLayer)
        print("Waited: ", y_check[i], " Get: ",  final)