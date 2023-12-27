import sys
import pandas as pd
import numpy as np
import argparse

from Sigmoid import Sigmoid
from Softmax import Softmax
from parsing import *
from Network import Network
from plotting import plot_curve

iteration = 10000
batch_size = 50
node_per_layer = 30
learning_rate = 0.0025
nb_feature = 30

def binaryCrossEntropy(output, y_train):
    n = y_train.shape[0]
    
    sum = 0
    for i in range(n):
        pred = output[i][0]
        sum += y_train[i] * np.log(pred) + ((1 - y_train[i]) * np.log(1 - pred))
    return sum  / n * -1

def main(data_path: str):

    try:
        data = pd.read_csv(data_path, header=None)
    except:
        print("Parsing error.") 
        exit(1)
    # random.seed(75)
    # np.random.seed(75)
    net = Network()
    net.addLayers(Sigmoid(node_per_layer, nb_feature, learning_rate))
    net.addLayers(Sigmoid(node_per_layer, node_per_layer, learning_rate))
    net.addLayers(Sigmoid(node_per_layer, node_per_layer, learning_rate))
    net.addLayers(Softmax(2, node_per_layer, learning_rate))

    data_y = data.drop(0, axis=1)
    data_x = data_y.drop(1, axis=1)
    data_y = data_y[1].replace('M', 1).replace('B', 0)

    epoch = iteration / (int(data_x.shape[0] / batch_size) + 1)
    epoch_itr = int(iteration / epoch)
    epoch_scaling = epoch / iteration

    historic_loss = np.zeros((int(epoch)+1, 3))
    for j in range(iteration):
        x_train, y_train, x_valid, y_valid = init_data(data_x, data_y, batch_size)
        final = net.feedforward(x_train, True)

        if j % epoch_itr == 0:
            loss = binaryCrossEntropy(final, y_train)
            pred = net.feedforward(x_valid, False)
            val_loss = binaryCrossEntropy(pred, y_valid)
            curr_idx = j * epoch_scaling
            historic_loss[int(curr_idx)] = [int(curr_idx), loss, val_loss]
            
            print("epoch: {0}/{1} - training_loss: {2} - validation_loss: {3}".format(int(curr_idx), int(epoch), round(loss, 4), round(val_loss, 4)))
        
        net.backpropagation(y_train)
    # plot_curve(historic_loss[:, 0], historic_loss[:, 1], historic_loss[:, 2])


    sum = 0
    x_accurany, y_accuracy, x_valid, y_valid = init_data(data_x, data_y, data_x.shape[0])

    final = net.feedforward(x_accurany, False)
    for k in range(final.shape[0]):
        # print("Waited: ", y_train[k], " Get: ",  final[k])
        if y_accuracy[k] == 1:
            sum += y_accuracy[k] - final[k][0]
        else:
            sum += final[k][0]
    precision = sum / final.shape[0]
    print("Accuracy: ", precision, " as ", (1 - precision) * 100, "%")



if __name__ == "__main__":
    params = argparse.ArgumentParser()
    params.add_argument("--data_path")
    args = params.parse_args()

    main(args.data_path)