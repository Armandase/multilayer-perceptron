import numpy as np

def accuracy(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    return np.mean(y_pred == y_true)

def binary_cross_entropy(y_true, y_pred):
    y_pred = np.argmax(y_pred, axis=1)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def derivative_binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return - (y_true / y_pred) + (1 - y_true) / (1 - y_pred)

#E = − 1/N N∑n=1[yn log(pn) + (1 − yn) log(1 − pn)]
def subject_binary_cross_entropy(y_true, y_pred):
    n = y_true.shape[0]
    
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    sum = np.sum(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return -1/n * sum

def derivative_subject_binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred))

def meanSquareError(y_train, y_pred):
    n = y_train.shape[0]

    diff = y_pred - y_train[:, 0]
    sum = np.sum(diff ** 2)
    return sum / n

def derivate_mean_square_error(y_train, y_pred):
    n = y_train.shape[0]
    diff = y_train - y_pred
    print(diff.shape)
    return -2 * diff / n
