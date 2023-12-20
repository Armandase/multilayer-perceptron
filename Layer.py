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
        self.learning_rate = 0.0001
        self.input = None
        self.output = None

    def computeLayer(self, input):
        self.input = input
    
        Z = np.dot(input, self.weights) + self.bias
        self.output = self.activation(Z)
        return self.output

    def deltaOutputLayer(self, waited_output):
        one_hot_y_train = one_hot(waited_output)
        delta = self.output - one_hot_y_train

        res = np.dot(delta, self.weights.T)
        # delta_bias = np.sum(delta, axis=0)
        # print(delta_bias)
        return res, delta
    
    # call this function in reverse order.
    def deltaHiddenLayer(self, above_delta):
        delta = above_delta * self.output * (1 - self.output) 
        res = np.dot(delta, self.weights.T)

        return res, delta

    # call this function in right order.
    def update_weights(self, delta, input):
        # delta_weights = np.dot(delta.T, self.output)
        delta_weights = np.dot(input.T, delta)
        # delta_weights = np.dot(delta.T, self.output)
        delta_weights = self.learning_rate * delta_weights
        self.weights = self.weights - delta_weights
