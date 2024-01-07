import json
import numpy as np
import argparse
import pandas as pd

from Network import Network
from Sigmoid import Sigmoid
from Softmax import Softmax

weight_path='save_model/model.json'
learning_rate = 0.0025

def main(data_path: str):
    try:
        data = pd.read_csv(data_path, header=None)
        predict(data)
    except Exception as e:
        print("An error occurred:", str(e))
        exit(1)

def predict(data):
    with open(weight_path, 'r', newline='') as infile:
        model = json.loads(infile.read())

        epoch = model['epoch']
        batch_size = model['batch_size']
        learning_rate = model['learning_rate']
        net = Network(learning_rate)

        for layer in model['network']:
            if layer['name'] == "sigmoid":
                net.addLayers(Sigmoid(0,0,1, np.array(layer['weights'])))
            elif layer['name'] == "softmax":
                net.addLayers(Softmax(0,0,1, np.array(layer['weights'])))
            else:
                raise Exception("Wrong layer name")

        iteration = epoch * (int(data.shape[0] / batch_size) + 1)
        print(iteration)

    
if __name__ == "__main__":
    params = argparse.ArgumentParser()
    params.add_argument("--data_path")
    args = params.parse_args()

    main(args.data_path)