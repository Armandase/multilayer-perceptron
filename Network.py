import json
import numpy as np
import math

from Layer import Layer
from constants import *
from parsing import *
from math_func import *

class Network:
    def __init__(self, learning_rate, batch_size=BATCH_SIZE, epochs=EPOCHS):
        self.layers = list()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs

    def addLayers(self, layer):
        if isinstance(layer, Layer) == False:
            print("addLayers: Wrong type of layer")
        
        if len(self.layers) > 0 and layer.weights.shape[0] != self.layers[-1].weights.shape[1]:
            print("addLayers: input len of new layers doesn't match (fully connected neural network)")

        layer.learning_rate = self.learning_rate
        self.layers.append(layer)

    def feedforward(self, input, train):
        X = input

        for layer in self.layers:
            X = layer.feedforward(X, train)
        return X

    def backpropagation(self, y_train):
        gradient = y_train

        for layer in reversed(self.layers):
            gradient = layer.backpropagation(gradient)

    def train_network(self, iteration, data_x, data_y):
        epoch_itr = int(iteration / self.epochs)
        epoch_scaling = self.epochs / iteration

        historic = np.empty((math.ceil(self.epochs), 5))
        for j in range(iteration):
            x_train, y_train, x_valid, y_valid = init_data(data_x, data_y, self.batch_size)
            final = self.feedforward(x_train, True)

            if j % epoch_itr == 0:
                loss = binaryCrossEntropy(final, y_train)
                accu = accuracy(y_train, final)
                
                pred = self.feedforward(x_valid, False)
                val_loss = binaryCrossEntropy(pred, y_valid)
                val_accu = accuracy(y_valid, pred)
                
                curr_idx = j * epoch_scaling
                historic[int(curr_idx)] = [int(curr_idx), loss, val_loss, accu, val_accu]
                
                print("epoch: {0}/{1}\n\
                        \ttraining loss: {2} - validation loss: {3}\n\
                        \ttraining accuracy: {4} - validation accuracy: {5}"
                    .format(int(curr_idx), int(self.epochs), round(loss, 4), round(val_loss, 4), round(accu, 4), round(val_accu, 4)))
            
            self.backpropagation(y_train)
        return historic

    def save_weights(self, batch_size=BATCH_SIZE, epoch=EPOCHS):
        data = {
            'learning_rate': self.learning_rate,
            'batch_size': batch_size,
            'epoch': epoch,
            'network': []}
        for layer in self.layers:
            layer_data = {
                'name': layer.name,
                'shape': [layer.weights.shape[0], layer.weights.shape[1]],
                'weights': layer.weights.tolist(),
                'bias': layer.bias.tolist(),
            }
            data['network'].append(layer_data)

        json_data = json.dumps(data, indent=4)
 
        try:
            with open(WEIGHT_PATH, 'w', newline='') as outfile:
                outfile.write(json_data)
            print('Written to file successfully')
        except Exception as e:
            print("An error occurred:", str(e))