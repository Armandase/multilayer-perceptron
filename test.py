import numpy as np

class NeuralNetwork:
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        # Initialize parameters with Xavier initialization
        self.W1 = np.random.randn(input_size, hidden_size1) / np.sqrt(input_size)
        self.b1 = np.zeros(hidden_size1)
        self.W2 = np.random.randn(hidden_size1, hidden_size2) / np.sqrt(hidden_size1)
        self.b2 = np.zeros(hidden_size2)
        self.W3 = np.random.randn(hidden_size2, output_size) / np.sqrt(hidden_size2)
        self.b3 = np.zeros(output_size)
    
    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        return self.sigmoid(x) * (1 - self.sigmoid(x))
    
    def softmax(self, x):
        exps = np.exp(x - np.max(x, axis=1, keepdims=True))  # Numerical stability
        return exps / np.sum(exps, axis=1, keepdims=True)
    
    def forward(self, X):
        # Layer 1
        self.a1 = np.dot(X, self.W1) + self.b1
        self.h1 = self.sigmoid(self.a1)
        # Layer 2
        self.a2 = np.dot(self.h1, self.W2) + self.b2
        self.h2 = self.sigmoid(self.a2)
        # Output layer
        self.a3 = np.dot(self.h2, self.W3) + self.b3
        self.output = self.softmax(self.a3)
        return self.output
    
    def backward(self, X, y_true, learning_rate):
        m = X.shape[0]
        
        # Output layer gradients
        d_a3 = (self.output - y_true) / m
        d_W3 = np.dot(self.h2.T, d_a3)
        d_b3 = np.sum(d_a3, axis=0)
        
        # Hidden layer 2 gradients
        d_h2 = np.dot(d_a3, self.W3.T)
        d_a2 = d_h2 * self.sigmoid_derivative(self.a2)
        d_W2 = np.dot(self.h1.T, d_a2)
        d_b2 = np.sum(d_a2, axis=0)
        
        # Hidden layer 1 gradients
        d_h1 = np.dot(d_a2, self.W2.T)
        d_a1 = d_h1 * self.sigmoid_derivative(self.a1)
        d_W1 = np.dot(X.T, d_a1)
        d_b1 = np.sum(d_a1, axis=0)
        
        # Update parameters
        self.W3 -= learning_rate * d_W3
        self.b3 -= learning_rate * d_b3
        self.W2 -= learning_rate * d_W2
        self.b2 -= learning_rate * d_b2
        self.W1 -= learning_rate * d_W1
        self.b1 -= learning_rate * d_b1

# Example usage:
def train(X_train, y_train, X_test, y_test, epochs, learning_rate):
    from sklearn.datasets import make_moons
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import OneHotEncoder
    from math_func import subject_binary_cross_entropy

    # Initialize network
    nn = NeuralNetwork(input_size=30, hidden_size1=16, hidden_size2=8, output_size=2)
    y_train = OneHotEncoder().fit_transform(y_train.reshape(-1, 1)).toarray()
    epochs = 10000
    # Training loop
    for epoch in range(epochs):
        # Forward pass and loss
        outputs = nn.forward(X_train)
        # loss = nn.compute_loss(y_train)
        loss = subject_binary_cross_entropy(y_train, outputs)
        
        # Backward pass and parameter update
        nn.backward(X_train, y_train, learning_rate)
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss:.4f}")

    # Test the model
    test_outputs = nn.forward(X_test)
    predicted_labels = np.argmax(test_outputs, axis=-1)
    true_labels = np.argmax(y_test)
    y_test = OneHotEncoder().fit_transform(y_test.reshape(-1, 1)).toarray()
    accuracy = np.mean(predicted_labels == true_labels)
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"valid loss: {subject_binary_cross_entropy(y_test, test_outputs)}")

def main(config_path: str):
    import os
    import logging
    import yaml
    from parsing import preprocessing 

    if config_path is None or os.path.exists(config_path) is False:
        logging.error(f"Invalid or missing config file: {config_path}")
        return
    
    verbose = False
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
        logging.info(f"Config loaded from {config_path}")

        verbose = config['verbose']
        preprocessing_config = config['preprocessing']
        train_x, train_y, test_x, test_y = preprocessing(preprocessing_config, verbose)

    config_model = config['model']
    batch_size = config_model['batch_size']
    epochs = config_model['epochs']
    learning_rate = config_model['learning_rate']
    train(train_x, train_y, test_x, test_y, epochs, learning_rate)

if __name__ == '__main__':
    import argparse
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', '-c', type=str, default='config.yaml')
    args = argparser.parse_args()
    # try:
    main(args.config)
    # except Exception as e:
        # print('Error:', e)

