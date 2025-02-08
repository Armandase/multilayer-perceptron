import json
import numpy as np
import argparse
import pandas as pd
import random
import os
import yaml

from parsing import one_hot
from Network import Network
from Sigmoid import Sigmoid
from Softmax import Softmax
from Dropout import Dropout
from Relu import Relu
from Tanh import Tanh
from parsing import load_dataset
from math_func import accuracy, binary_cross_entropy, subject_binary_cross_entropy, meanSquareError

def predict(data_x, data_y, model_path, seed):
    with open(model_path, 'r', newline='') as infile:
        if seed != -1:
            random.seed(seed)
            np.random.seed(seed)
        model = json.loads(infile.read())

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
            elif layer['name'] == "tanh":
                net.addLayers(Tanh(weights=weights, bias=bias))
            else:
                raise Exception("Wrong layer name")

        y_pred = net.feedforward(data_x, False)

        precision = accuracy(data_y, y_pred)
        print("Accuracy: ", precision, " as ", (1 - precision) * 100, "%")
        
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


    data_x, data_y = load_dataset(predict_config['data_path'], config['preprocessing'], verbose)
    
    predict(data_x, data_y, predict_config['model_path'], predict_config['seed'])

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', '-c', type=str, default='config.yaml')
    args = argparser.parse_args()
    # try:
    main(args.config)
    # except Exception as e:
        # print('Error:', e)