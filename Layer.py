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
        self.aggregation = None

    def derivate(self, Z):
        return self.activation(Z) * (1 - self.activation(Z))

    def computeLayer(self, input):
        self.input = input
    
        self.aggregation = np.dot(input, self.weights) + self.bias
        self.output = self.activation(self.aggregation)
        return self.output

    def deltaOutputLayer(self, waited_output):
        one_hot_y_train = one_hot(waited_output)
        delta = self.output - one_hot_y_train
        # delta = self.derivate(self.aggregation) * delta
        res = np.dot(delta, self.weights.T)
        return res
    
    # call this function in reverse order.
    def deltaHiddenLayer(self, above_delta):
        delta = above_delta * ((1 - self.output) * self.output)
        # delta = self.derivate(self.aggregation) * above_delta
        res = np.dot(delta, self.weights.T)
        return res

    # call this function in right order.
    def update_weights(self, delta):
        delta_weights = np.dot(delta.T, self.output)
        # delta_weights = np.dot(self.output.T, delta)
        delta_weights = self.learning_rate * delta_weights
        self.weights = self.weights - delta_weights
