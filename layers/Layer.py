import numpy as np
import random
from parsing import *
from abc import ABC, abstractmethod
from AdamOptim import AdamOptim
from RMSprop import RMSprop

def initialize_weights(input_len, output_len, method="he_uniform"):
    if method == "random_default":
        return np.random.rand((input_len, output_len))
    elif method == "random_uniform":
        return np.random.uniform(-1, 1, (input_len, output_len))
    elif method == "random_normal":
        return np.random.normal(0, 1, (input_len, output_len))
    elif method == "he_normal":
        return np.random.normal(0, np.sqrt(2/input_len), (input_len, output_len))
    elif method == "he_uniform":
        return np.random.uniform(-np.sqrt(6/input_len), np.sqrt(6/input_len), (input_len, output_len))  
    else:
        raise ValueError("Invalid method for initializing weights")
    
def optimizers(name):
    if name == "adam":
        return AdamOptim
    elif name == "rmsprop":
        return RMSprop
    elif name is None or name == "":
        return None
    raise ValueError("Invalid optimizer")

class Layer(ABC):
    def __init__(self, input_len=0, output_len=0,
                 learning_rate=0.01, weights=None, bias=None,
                 dropout_rate=0.3, optimizer=None):
        if weights is not None:
            self.weights = weights
        if bias is not None:
            self.bias = bias
        else:
            self.bias = np.zeros(output_len)
            self.weights = initialize_weights(input_len, output_len)
        self.learning_rate = learning_rate
        self.input = None
        self.output = None
        self.weights_grad = None
        self.bias_grad = None
        self.name = ""
        self.set_name()
        if optimizer is not None:
            self.optimizer = optimizers(optimizer)(eta=learning_rate)
        else:
            self.optimizer = None
        if self.name == "dropout":
            if dropout_rate > 1 or dropout_rate < 0:
                raise Exception("Dropout rate should be between 0 and 1.")
            self.dropout_rate = dropout_rate

    def __str__(self):
        return self.name + " Layer: input " + str(self.weights.shape[0]) + " output " + str(self.weights.shape[1])
    
    @abstractmethod
    def set_name(self):
        pass

    @abstractmethod
    def activation_function(self, Z):
        pass

    @abstractmethod
    def derivative_activation_function(self, Z):
        pass

    @abstractmethod
    def backpropagation(self):
        pass

    def feedforward(self, input, train):
        weighted_sums = np.dot(input, self.weights) + self.bias
        output = self.activation_function(weighted_sums)

        if train:
            self.input = input
            self.output = weighted_sums
        return output
    
    def upate_weights(self, epoch):
        if self.weights_grad is None or self.bias_grad is None:
            return
        if self.optimizer is None:
            self.weights -= self.learning_rate * self.weights_grad
            self.bias -= self.learning_rate * self.bias_grad
        else:
            self.weights, self.bias = self.optimizer.update(epoch, self.weights, self.bias, self.weights_grad, self.bias_grad)