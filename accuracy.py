import numpy as np

from parsing import init_data

def compute_accuracy(data_x, data_y, net, y_train):
    #collect x and y normalized dataset without store valids datasets
    x_accurany, y_accuracy = init_data(data_x, data_y, data_x.shape[0])[:2]
    final = net.feedforward(x_accurany, False)

    #final[:, 0] same as final.T[0]
    diff = y_accuracy - final[:, 0]
    sum = np.sum(np.abs(diff))

    precision = sum / final.shape[0]
    print("Accuracy: ", precision, " as ", (1 - precision) * 100, "%")
    return precision