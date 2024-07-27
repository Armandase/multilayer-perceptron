import numpy as np

from parsing import init_data


def accuracy(y_true, y_pred):
    #based on shape like (n, 2) and (n, 1)
    y_pred = np.argmax(y_pred, axis=1)
    return np.mean(y_pred == y_true)


def binary_cross_entropy(y_true, y_pred):
    # epsilon = 1e-15
    # y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    # return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    y_true_int = y_true.astype(int)
    return -np.mean(np.log(y_pred[np.arange(len(y_pred)), y_true_int]))

def binary_cross_entropy_derivative(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.shape[0])



def meanSquareError(y_pred, y_train):
    n = y_train.shape[0]

    diff = y_train - y_pred[:, 0]
    sum = np.sum(diff ** 2)
    return sum / n
