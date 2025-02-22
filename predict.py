import json
import numpy as np
import argparse
import pandas as pd
import random
import os
import yaml

from parsing import one_hot, normalize_data_mean_std
from Network import Network
from layers.Sigmoid import Sigmoid
from layers.Softmax import Softmax
from layers.Dropout import Dropout
from layers.Relu import Relu
from layers.Tanh import Tanh
from layers.BatchNorm import BatchNorm
from layers.L1Norm import L1Normalization
from parsing import load_dataset
from math_func import accuracy, binary_cross_entropy, subject_binary_cross_entropy, meanSquareError

def predict(data_x, data_y, model_path, seed):
    with open(model_path, 'r', newline='') as infile:
        if seed != -1:
            random.seed(seed)
            np.random.seed(seed)
        model = json.loads(infile.read())

        mean = model['mean']
        std = model['std']

        data_x
        data_x, _ ,_ = normalize_data_mean_std(data_x, mean, std)
        net = Network()

        for layer in model['network']:
            weights = np.array(layer['weights'])
            bias = np.array(layer['bias'])
            if layer['name'] == "sigmoid":
                net.addLayers(Sigmoid(weights=weights, bias=bias))
            elif layer['name'] == "relu":
                net.addLayers(Relu(weights=weights, bias=bias))
            elif layer['name'] == "softmax":
                net.addLayers(Softmax(weights=weights, bias=bias))
            elif layer['name'] == "dropout":
                net.addLayers(Dropout(weights=weights, bias=bias))
            elif layer['name'] == "batchnorm":
                net.addLayers(BatchNorm(weights=weights, bias=bias))
            elif layer['name'] == "l1_normalization":
                net.addLayers(L1Normalization(weights=weights, bias=bias))
            elif layer['name'] == "tanh":
                net.addLayers(Tanh(weights=weights, bias=bias))
            else:
                raise Exception("Wrong layer name")

        y_pred = net.feedforward(data_x, train=False)

        precision = accuracy(data_y, y_pred)
        print("Accuracy: ", precision, " also ", (precision * 100), "%")
        
        loss = binary_cross_entropy(data_y, y_pred)
        print("Loss global: ", loss)

        loss_subject = subject_binary_cross_entropy(one_hot(data_y), y_pred)
        print("Loss subject: ", loss_subject)

        loss_mse = meanSquareError(y_pred, data_y)
        print("Loss MSE: ", loss_mse)

def main(config_path: str):
    if config_path is None or os.path.exists(config_path) is False:
        print(f"Invalid or missing config file: {config_path}")
        return
    
    verbose = False
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
        print(f"Config loaded from {config_path}")

        verbose = config['verbose']
        predict_config = config['predict']


    data_x, data_y = load_dataset(predict_config['data_path'], config['preprocessing'], verbose, remove_outliers=False)
    
    predict(data_x, data_y, predict_config['model_path'], predict_config['seed'])

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', '-c', type=str, default='config.yaml')
    args = argparser.parse_args()
    try:
        main(args.config)
    except Exception as e:
        print('Error:', e)