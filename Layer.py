import numpy as np

class Layer:
    def __init__(self, nodes, input_len ,funcActivation):
        self.nodes = nodes
        self.bias = np.zeros(nodes)
        self.activation = funcActivation
        self.weights = np.random.rand(input_len, nodes)
        self.learning_rate = 1

    def computeLayer(self, input):
        # Z1 = self.weights.dot(input) + self.bias
        Z1 = input.dot(self.weights)
        output = self.activation(Z1)
        return output

    def deltaOutputLayer(self, waited_output, output):
        delta = output * (1 - output) * (waited_output - output)
        res = np.dot(delta, self.weights.T)
        return res
    
    # call this function in reverse order.
    def deltaHiddenLayer(self, above_delta, output, input):
        delta = above_delta * (1 - output) * output
        res = np.dot(delta, self.weights.T)
        return res

    # call this function in right order.
    def update_weights(self, output, delta):
        # print("UPDATE WEIGHTS: ", output.T.shape, " * ", delta.shape)

        delta_weights = np.dot(output.T, delta)
        delta_weights = self.learning_rate * delta_weights
        self.weights = self.weights - delta_weights.T
