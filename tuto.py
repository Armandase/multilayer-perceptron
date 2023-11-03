nodes = 10

def ReLu (Z):
    return np.maximum(0, Z)

def softmax(Z):
    expZ = np.exp(Z)
    return expZ / np.sum(expZ)

def forward_prop(W1, b1, W2, b2, x_train):
    Z1 = W1.dot(x_train) + b1
    A1 = ReLu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

def one_hot(y_train):
    one_hot = np.zeros((y_train.size, y_train.max() + 1))
    one_hot[np.arange(y_train.size, y_train)] = 1
    return one_hot.T

def back_prop(Z1, A1, Z2, A2, W2, y_train):
    Z1 = W1.dot(x_train) + b1
    A1 = ReLu(Z1)
    Z2 = W2.dot(A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2

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
    
    W1 = np.random.rand(nodes, input_len)
    b1 = np.random.rand(nodes, 1)
    W2 = np.random.rand(nodes, 2)
    b2 = np.random.rand(nodes, 1)


    
if __name__ == "__main__":
    main()
    