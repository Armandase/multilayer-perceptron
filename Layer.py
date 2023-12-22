import numpy as np
import random
from parsing import *

class Layer:
    def __init__(self, nodes, input_len ,funcActivation, learning_rate):
        self.bias = np.zeros(nodes)
        self.activation = funcActivation
        self.weights = np.random.rand(input_len, nodes) / np.sqrt(input_len)
        rng = np.random.default_rng()
        self.learning_rate = learning_rate
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

        delta_weights = np.dot(self.input.T, delta) * self.learning_rate
        self.weights = self.weights - delta_weights

        delta_bias = np.sum(delta, axis=0)
        self.bias = self.bias - delta_bias * self.learning_rate

        res = np.dot(delta, self.weights.T)
        return res
    
    def deltaHiddenLayer(self, above_delta):
        delta = above_delta * self.output * (1 - self.output) 

        delta_weights = np.dot(self.input.T, delta) * self.learning_rate
        self.weights = self.weights - delta_weights

        delta_bias = np.sum(delta, axis=0)
        self.bias = self.bias - delta_bias * self.learning_rate

        res = np.dot(delta, self.weights.T)
        return res