import pandas as pd
import numpy as np
import argparse
import math

from Sigmoid import Sigmoid
from Softmax import Softmax
from parsing import *
from math_func import *
from Network import Network
from plotting import plot_curve
from constants import *

def main(data_path: str, epochs: int, batch_size: int, learning_rate: float):

    try:
        data = pd.read_csv(data_path, header=None)
    except:
        print("Parsing error.") 
        exit(1)
    random.seed(SEED)
    np.random.seed(SEED)
    net = Network(learning_rate)
    net.addLayers(Sigmoid(NODE_PER_LAYER, NB_FEATURE))
    net.addLayers(Sigmoid(NODE_PER_LAYER, NODE_PER_LAYER))
    net.addLayers(Sigmoid(NODE_PER_LAYER, NODE_PER_LAYER))
    net.addLayers(Softmax(NB_OUTPUT, NODE_PER_LAYER))

    data_y = data.drop(0, axis=1)
    data_x = data_y.drop(1, axis=1)
    data_y = data_y[1].replace('M', 1).replace('B', 0)

    iteration = int(epochs * (int(data_x.shape[0] / batch_size) + 1))
    
    epoch_itr = int(iteration / epochs)
    epoch_scaling = epochs / iteration

    historic_loss = np.empty((math.ceil(epochs), 5))
    for j in range(iteration):
        x_train, y_train, x_valid, y_valid = init_data(data_x, data_y, batch_size)
        final = net.feedforward(x_train, True)

        if j % epoch_itr == 0:
            loss = binaryCrossEntropy(final, y_train)
            accu = accuracy(y_train, final)
            
            pred = net.feedforward(x_valid, False)
            val_loss = binaryCrossEntropy(pred, y_valid)
            val_accu = accuracy(y_valid, pred)
            
            curr_idx = j * epoch_scaling
            historic_loss[int(curr_idx)] = [int(curr_idx), loss, val_loss, accu, val_accu]
            
            print("epoch: {0}/{1}\n\
                    \ttraining loss: {2} - validation loss: {3}\n\
                    \ttraining accuracy: {4} - validation accuracy: {5}"
                  .format(int(curr_idx), int(epochs), round(loss, 4), round(val_loss, 4), round(accu, 4), round(val_accu, 4)))
        
        net.backpropagation(y_train)
    
    plot_curve(historic_loss[:, 0], historic_loss[:, 1], historic_loss[:, 2], historic_loss[:, 3], historic_loss[:, 4])
    net.save_weights()

if __name__ == "__main__":
    params = argparse.ArgumentParser()
    params.add_argument("--data_path", help="Path to the data")
    params.add_argument("--epochs", help="Epoch refers to one complete pass through the dataset during the training phase")
    params.add_argument("--learning_rate", help="Learning rate controls how quickly the model adjusts its weights ")
    params.add_argument("--batch_size", help="Batch size determines the number of data samples used in a single iteration.")
    args = params.parse_args()    
    
    data_path, epochs, batch_size, learning_rate = handle_args(args)
    
    main(data_path, epochs, batch_size, learning_rate)