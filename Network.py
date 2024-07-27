import json
import numpy as np
import math

from Layer import Layer
from parsing import *
from math_func import *

class Network:
    def __init__(self):
        self.layers = []

    def addLayers(self, layer):
        if isinstance(layer, Layer) == False:
            print("addLayers: Wrong type of layer")
            return 
        
        if len(self.layers) > 0 and layer.weights.shape[0] != self.layers[-1].weights.shape[1]:
            print("addLayers: input len of new layers doesn't match (fully connected neural network)")
            return

        self.layers.append(layer)

    def feedforward(self, input, train):
        output = input

        for layer in self.layers:
            output = layer.feedforward(output, train)
        return output

    # def backpropagation(self, y_train):
    #     gradient = y_train
    #     for layer in reversed(self.layers):
    #         gradient = layer.backpropagation(gradient)

    #     for layer in self.layers:
    #         layer.upate_weights()        

    def backpropagation(self, y_train):
        pred = self.layers[-1].output
        error = pred
        error[np.arange(len(pred)), y_train.astype(np.int64) ] -= 1
        error /= len(pred)
        # print('error shape:', error.shape)
        gradients = []

        # Iterate over each layer in reverse order
        for i in range(len(self.layers) - 1, -1, -1):
            # Compute the gradients of the loss with respect to the weights and biases
            gradients.append([np.dot(self.layers[i].input.T, error)])
            gradients[-1].append(np.mean(error, axis=0))
            
            # Compute the error of the current layer
            if i >= 1:  # avoid index error
                error = np.dot(error, self.layers[i].weights.T)
                error *= self.layers[i - 1].derivative_activation_function(self.layers[i].input)
        # exit()
        # Reverse the list of gradients to match the order of the layers
        gradients.reverse()

        self.gradients = gradients
        for i in range(len(self.layers)):
            # Update the weights and biases of the current layer
            # print(i, ' with', self.layers[i].weights.shape, ' * ', self.gradients[i][0].shape)
            self.layers[i].weights -= self.learning_rate * self.gradients[i][0]
            self.layers[i].bias -= self.learning_rate * self.gradients[i][1]
        # exit()

    def fit(self, batch_size, epochs, learning_rate, data_x, data_y):
        self.learning_rate = learning_rate
        for epoch in range(epochs):
            batches_x, batches_y = get_batches(data_x, data_y, batch_size)

            for x, y in zip(batches_x, batches_y):
                y_one_hot = np.zeros((y.size, y.max()+1))
                y_one_hot[np.arange(y.size), y] = 1
                a = np.array(y, copy=True)

                output = self.feedforward(x, True)
                zob = np.array(output, copy=True)
                
                loss_entropy = binary_cross_entropy(y, output)
                self.backpropagation(y)
                # loss_entropy = binaryCrossEntropy(output, y)
                # loss_mse = meanSquareError(output, y)
                accu = accuracy(a, zob)
            print("epoch: {0}/{1}\n\
                    \ttraining loss entropy: {2}\n\
                    \ttraining accuracy: {3}"
                .format(epoch, epochs, round(loss_entropy, 4), round(accu, 4)))
                
            # pred = self.feedforward(x_valid, False)
            # val_loss_entropy = binaryCrossEntropy(pred, y_valid)
            # val_loss_mse = meanSquareError(pred, y_valid)
            # val_accu = accuracy(y_valid, pred)
            
            # curr_idx = j * epoch_scaling
            # historic[int(curr_idx)] = [int(curr_idx), accu, val_accu, loss_entropy, val_loss_entropy, loss_mse, val_loss_mse]
            
            # print("epoch: {0}/{1}\n\
            #         \ttraining loss entropy: {2} - validation loss entropy: {3}\n\
            #         \ttraining accuracy: {4} - validation accuracy: {5}"
            #     .format(int(curr_idx), int(self.epochs), round(loss_entropy, 4), round(val_loss_entropy, 4), round(accu, 4), round(val_accu, 4)))
        
            # if val_loss_entropy < EARLY_STOP:
            #     break
            
            # self.backpropagation(y_train)
        return 

    # def save_weights(self, batch_size=BATCH_SIZE, epoch=EPOCHS):
    #     data = {
    #         'learning_rate': self.learning_rate,
    #         'batch_size': batch_size,
    #         'epoch': epoch,
    #         'network': []}
    #     for layer in self.layers:
    #         layer_data = {
    #             'name': layer.name,
    #             'shape': [layer.weights.shape[0], layer.weights.shape[1]],
    #             'weights': layer.weights.tolist(),
    #             'bias': layer.bias.tolist(),
    #         }
    #         data['network'].append(layer_data)

    #     json_data = json.dumps(data, indent=4)
 
    #     try:
    #         with open(WEIGHT_PATH, 'w', newline='') as outfile:
    #             outfile.write(json_data)
    #         print('Written to file successfully')
    #     except Exception as e:
    #         print("An error occurred:", str(e))