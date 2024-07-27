import numpy as np
import random
from parsing import *


class Layer:
    def __init__(self, input_len=0, output_len=0, learning_rate=0.01, weights=None, bias=None):
        if weights and bias:
            self.weights = weights
            self.bias = bias
        else:
            self.bias = np.zeros(output_len)
            self.weights = np.random.rand(input_len, output_len) / np.sqrt(input_len)
        self.learning_rate = learning_rate
        self.input = None
        self.output = None
        self.delta = None
        self.set_name()

    def feedforward(self, input, train):
        weighted_sums = np.dot(input, self.weights) + self.bias
        output = self.activation_function(weighted_sums)

        if train:
            self.input = input
            self.output = output
        return output
    
    def upate_weights(self):
        # delta_weights = np.dot(self.input.T, self.delta) * self.learning_rate
        delta_weights = np.dot(self.input.T, self.delta)

        self.weights = self.weights - delta_weights * self.learning_rate

        delta_bias = np.sum(self.delta, axis=0)
        self.bias = self.bias - delta_bias * self.learning_rate
