import numpy as np
import random
from parsing import *

class Layer:
    def __init__(self, nodes, input_len, learning_rate):
        self.bias = np.zeros(nodes)
        self.weights = np.random.rand(input_len, nodes) / np.sqrt(input_len)
        # rng = np.random.default_rng()
        self.learning_rate = learning_rate
        self.input = None
        self.output = None

    def feedforward(self, input, train):
    
        Z = np.dot(input, self.weights) + self.bias
        output = self.activation_function(Z)

        if train == True:
            self.input = input
            self.output = output
        return output