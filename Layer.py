import numpy as np
import random
from parsing import *

class Layer:
    def __init__(self, nodes=0, input_len=0, learning_rate=1, weights=None):
        if weights is None:
            self.bias = np.zeros(nodes)
            self.weights = np.random.rand(input_len, nodes) / np.sqrt(input_len)
        else:
            self.weights = weights
            self.bias = weights.shape[1]
        self.learning_rate = learning_rate
        self.input = None
        self.output = None
        self.set_name()

    def feedforward(self, input, train):
    
        Z = np.dot(input, self.weights) + self.bias
        output = self.activation_function(Z)

        if train == True:
            self.input = input
            self.output = output
        return output
