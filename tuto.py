import sys
import pandas as pd
import numpy as np

nodes = 2
lr = 1
iterations = 1000
percentage_from_data = 0.85


def ReLu (Z):
    return np.maximum(0, Z)

def deriv_ReLu (Z):
    return Z > 0

def softmax(Z):
    expZ = np.exp(Z)
    print(expZ / np.sum(expZ))
    return expZ / np.sum(expZ)

def forward_prop(W1, b1, W2, b2, x_train):
    Z1 = W1.dot(x_train) + b1
    A1 = ReLu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(y_train):
    one_hot = np.zeros((y_train.size, y_train.max() + 1))
    one_hot[np.arange(y_train.size), y_train] = 1
    return one_hot.T

def back_prop(Z1, A1, Z2, A2, W2, y_train, x_train):
    m = y_train.size
    one_hot_Y = one_hot(y_train)

    dZ2 = A2 - one_hot_Y
    dW2 = 1 / m * dZ2.dot(A1.T)
    db2 = 1 / m * np.sum(dZ2)

    dZ1 = W2.T.dot(dZ2) * deriv_ReLu(Z1)
    dW1 = 1 / m * dZ1.dot(X.T)
    db1 = 1 / m * np.sum(dZ1)
    return dW1, db1, dW2, db2

def update_params(dW1, db1, dW2, db2, W1, b1, W2, b2):
    W1 = W1 - lr * dW1
    b1 = b1 - lr * db1
    W2 = W2 - lr * dW2
    b2 = b2 - lr * db2
    return W1, b1, W2, b2

def gradient_descent(x_train, y_train):
    input_len = x_train.shape[0]
    W1 = np.random.rand(nodes, input_len)
    b1 = np.random.rand(nodes, 1)
    W2 = np.random.rand(nodes, nodes)
    b2 = np.random.rand(nodes, 1)

    for i in range(iterations):
        Z1, A1, Z2, A2 = forward_prop(W1, b1, W2, b2, x_train)
        dW1, db1, dW2, db2 = back_prop(Z1, A1, Z2, A2, W2, y_train, x_train)
        W1, b1, W2, b2 = update_params(dW1, db1, dW2, db2, W1, b1, W2, b2)
        if i % 100 == 0:
            print("iteration: ", i)
            print(Y - A2)
    return W1, b1, W2, b2

def main():
    data = pd.read_csv(sys.argv[1], header=None)
    data = data.drop(0, axis=1)
    
    # create y train (waited output of our neural network)
    y_train = data[1].copy()
    y_train = y_train[:round(data.shape[0] * percentage_from_data)]
    y_train = y_train.replace('M', 1)
    y_train = y_train.replace('B', 0)
    y_train = np.array(y_train)
    data = data.drop(1, axis=1)
    # initialize x values (input of our neural network)
    x_train = pd.DataFrame(data[0:round(data.shape[0] * percentage_from_data)].values)
    x_valid = pd.DataFrame(data[x_train.shape[0]:data.shape[0]].values)
    x_train = np.array(x_train)
    # x_train = normalize_data(x_train)

    gradient_descent(x_train, y_train)
    
if __name__ == "__main__":
    main()
    