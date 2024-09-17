import numpy as np
import random
from parsing import *
from abc import ABC, abstractmethod


class Layer(ABC):
    def __init__(self, input_len=0, output_len=0, learning_rate=0.01, weights=None, bias=None, dropout_rate=0.3):
        if weights is not None:
            self.weights = weights
        if bias is not None:
            self.bias = bias
        else:
            self.bias = np.zeros(output_len)
            self.weights = np.random.rand(input_len, output_len) / np.sqrt(input_len)
        self.learning_rate = learning_rate
        self.input = None
        self.output = None
        self.weights_grad = None
        self.bias_grad = None
        self.name = ""
        self.set_name()
        if self.name == "dropout":
            self.dropout_rate = dropout_rate

    
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
            self.output = output
        return output
    
    def upate_weights(self):
        if self.weights_grad is not None:
            self.weights -= self.learning_rate * self.weights_grad
        if self.bias_grad is not None:
            self.bias -= self.learning_rate * self.bias_grad
