import json
import numpy as np
import argparse
import pandas as pd
import random

from math_func import *
from Network import Network
from Sigmoid import Sigmoid
from Softmax import Softmax
from parsing import init_data
from constants import *

def predict(data_x, data_y):
    with open(WEIGHT_PATH, 'r', newline='') as infile:
        random.seed(SEED)
        np.random.seed(SEED)
        model = json.loads(infile.read())

        # epoch = model['epoch']
        # batch_size = model['batch_size']
        # iteration = epoch * (int(data_x.shape[0] / batch_size) + 1)
        learning_rate = model['learning_rate']
        net = Network(learning_rate)

        for layer in model['network']:
            weights = np.array(layer['weights'])
            bias = np.array(layer['bias'])
            if layer['name'] == "sigmoid":
                net.addLayers(Sigmoid(0,0,learning_rate, weights, bias))
            elif layer['name'] == "softmax":
                net.addLayers(Softmax(0,0,learning_rate, weights, bias))
            else:
                raise Exception("Wrong layer name")

        x_global, y_global = init_data(data_x, data_y, data_x.shape[0])[:2]
        y_pred = net.feedforward(x_global, False)

        precision = accuracy(y_global, y_pred)
        print("Accuracy: ", precision, " as ", (1 - precision) * 100, "%")
        
        loss = binaryCrossEntropy(y_pred, y_global)
        print("Loss global: ", loss)

def main(data_path: str):
    try:
        data = pd.read_csv(data_path, header=None)
        data_y = data.drop(0, axis=1)
        data_x = data_y.drop(1, axis=1)
        data_y = data_y[1].replace('M', 1).replace('B', 0)
        
        predict(data_x, data_y)
    except Exception as e:
        print("An error occurred:", str(e))
        exit(1)

if __name__ == "__main__":
    params = argparse.ArgumentParser()
    params.add_argument("--data_path")
    args = params.parse_args()

    main(args.data_path)