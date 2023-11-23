import numpy as np

class Layer:
    def __init__(self, nodes, input_len ,funcActivation):
        self.nodes = nodes
        self.bias = np.zeros((nodes, 1))
        self.activation = funcActivation
        self.weights = np.random.rand(nodes, input_len)
        self.learning_rate = 1

    def computeLayer(self, input):
        # Z1 = self.weights.dot(input) + self.bias.T
        Z1 = self.weights.dot(input)

        output = self.activation(Z1)
        return output  
    
    def update_params(dB, dW):
        self.weights = self.weights - learning_rate * dW
        self.bias = self.bias - learning_rate * dB


