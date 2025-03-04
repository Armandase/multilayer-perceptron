import json
import numpy as np
import prettytable

from layers.Layer import Layer
from parsing import *
from math_func import *
from Callback import Callback

def init_table(current_epoch, epochs):
    table = prettytable.PrettyTable()
    table.title = f"Epochs {current_epoch + 1}/{epochs}"
    table.field_names = ["Set used", "Binary Cross Entropy", "Subject Entropy", "Mean Square Entropy", "Accuracy"] 
    return table

def update_historic(historic, epoch, accu, val_accu, loss_entropy, val_loss_entropy, loss_mse, val_loss_mse):
    historic["epoch"] = epoch
    historic["accu"].append(accu)
    historic["val_accu"].append(val_accu)
    historic["loss_entropy"].append(loss_entropy)
    historic["val_loss_entropy"].append(val_loss_entropy)
    historic["loss_mse"].append(loss_mse)
    historic["val_loss_mse"].append(val_loss_mse)
    

class Network:
    def __init__(self, mean=None, std=None, callbacks=None, deriv_loss=None):
        self.layers = []
        self.callbacks = None
        self.derivative_loss = None
        self.mean = mean
        self.std = std
        if callbacks:
            self.callbacks = callbacks
        if deriv_loss:
            self.derivative_loss = deriv_loss

    def __str__(self): 
        str_layers = "Neural Network: \n"
        for layer in self.layers:
            str_layers += str(layer) + "\n"
        return str_layers

    def addLayers(self, layer):
        if isinstance(layer, Layer) == False:
            print("addLayers: Wrong type of layer")
            return 
        
        if len(self.layers) > 0 and layer.weights.shape[0] != self.layers[-1].weights.shape[1] and self.layers[-1].name != "dropout":
            print("addLayers: input len of new layers doesn't match (fully connected neural network)")
            return

        self.layers.append(layer)

    def feedforward(self, input, train):
        output = input

        for layer in self.layers:
            output = layer.feedforward(output, train)
        return output

    def update_model(self, epoch):
        for layer in self.layers:
            layer.upate_weights(epoch)

    def backpropagation(self, y_train, epoch):
        gradient = y_train
        for layer in reversed(self.layers):
            gradient = layer.backpropagation(gradient)
        self.update_model(epoch)

    def fit(self, train_x, train_y, test_x, test_y,
            batch_size=64, epochs=1000, learning_rate=0.01, 
            verbose=True, early_stopping=0.0001):
        self.early_stopping = early_stopping
        self.learning_rate = learning_rate

        historic = {
            "epoch": 0, "accu": [], "val_accu": [], "loss_entropy": [], "val_loss_entropy": [], "loss_mse": [], "val_loss_mse": []
        }

        best_subject_entropy = np.inf
        best_accuracy = 0
        for epoch in range(epochs):
            table = init_table(epoch, epochs)
            train_x, train_y = shuffle_data(train_x, train_y)
            batches_x, batches_y = get_batches(train_x, train_y, batch_size)
            prev_val_loss_entropy = np.inf            

            avg_accu = 0
            avg_loss_entropy = 0
            avg_mse = 0
            avg_subject_entropy = 0
            nb_batches = len(batches_x)
            for x, y in zip(batches_x, batches_y):
                y_one_hot = one_hot(y)
                output = self.feedforward(x, True)
                
                avg_loss_entropy += binary_cross_entropy(np.array(y, copy=True), np.array(output, copy=True))
                avg_mse += meanSquareError(output, y)
                avg_accu += accuracy(np.array(y, copy=True), np.array(output, copy=True))
                avg_subject_entropy += subject_binary_cross_entropy(np.array(y_one_hot, copy=True), np.array(output, copy=True))

                error = self.derivative_loss(np.array(y_one_hot, copy=True), np.array(output, copy=True))
                self.backpropagation(error, epoch)
                
            loss_entropy = avg_loss_entropy / nb_batches
            loss_subject_entropy = avg_subject_entropy / nb_batches
            loss_mse = avg_mse / nb_batches
            accu = avg_accu / nb_batches
            table.add_row(["Training", round(loss_entropy, 4), round(loss_subject_entropy, 4), round(loss_mse, 4), round(accu, 4)])

            # Compute test metrics
            pred = self.feedforward(test_x, False)
            val_loss_entropy = binary_cross_entropy(test_y, pred)
            val_loss_subject_entropy = subject_binary_cross_entropy(one_hot(test_y), pred)
            val_loss_mse = meanSquareError(pred, test_y)
            val_accu = accuracy(test_y, pred)

            if best_subject_entropy > val_loss_subject_entropy:
                best_subject_entropy = val_loss_subject_entropy
            if best_accuracy < val_accu:
                best_accuracy = val_accu

            table.add_row(["Validation", round(val_loss_entropy, 4), round(val_loss_subject_entropy, 4), round(val_loss_mse, 4), round(val_accu, 4)])
            print(table)
            
            update_historic(historic, epoch, accu, val_accu, loss_subject_entropy, val_loss_subject_entropy, loss_mse, val_loss_mse)
            
            if self.callbacks.callback_manager(historic, self) == True:
                break 

            if np.abs(prev_val_loss_entropy - val_loss_entropy) < self.early_stopping:
                break
            prev_val_loss_entropy = val_loss_entropy

        print(f"Best subject entropy with data_test: {best_subject_entropy:4f}")
        print(f"Best accuracy with data_test: {best_accuracy:4f}")
        return historic 

    def save_weights(self, path):
        if self.mean is None or self.std is None:
            raise Exception("Mean and std must be set before saving the model")
        data = {
            'mean': self.mean.tolist(),
            'std': self.std.tolist(),
            'network': []
        }

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
            with open(path, 'w', newline='') as outfile:
                outfile.write(json_data)
            print('Written to file successfully')
        except Exception as e:
            print("An error occurred:", str(e))