import numpy as np
import random
from parsing import *

class Layer:
    def __init__(self, nodes, input_len ,funcActivation):
        self.nodes = nodes
        self.bias = np.zeros(nodes)
        self.activation = funcActivation
        self.weights = np.random.rand(input_len, nodes) / np.sqrt(input_len)
        rng = np.random.default_rng()
        self.learning_rate = 1
        self.input = None
        self.output = None

    def computeLayer(self, input):
        self.input = np.array(input)
    
        Z1 = np.dot(input, self.weights) + self.bias
        self.output = np.array(self.activation(Z1))
        return self.output

    def deltaOutputLayer(self, waited_output):
        # delta = np.empty([self.output.shape[0], 2])
        # for i in range(self.output.shape[0]):
        #     if waited_output[i] == 0:
        #         delta[i] = np.array([self.output[i][0] - 1, self.output[i][1] - 0])
        #     else:
        #         delta[i] = np.array([self.output[i][0] - 0, self.output[i][1] - 1])

        one_hot_y_train = one_hot(waited_output)
        delta = self.output - one_hot_y_train
        res = np.dot(delta, self.weights.T)
        return res
    
    # call this function in reverse order.
    def deltaHiddenLayer(self, above_delta):
        delta = above_delta * (1 - self.output) * self.output
        res = np.dot(delta, self.weights.T)
        return res

    # call this function in right order.
    def update_weights(self, delta):
        delta_weights = np.dot(delta.T, self.output)
        # delta_weights = np.dot(self.output.T, delta)
        delta_weights = self.learning_rate * delta_weights
        self.weights = self.weights - delta_weights
