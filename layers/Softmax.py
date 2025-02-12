from layers.Layer import Layer
import numpy as np
from parsing import *
from math_func import *

class Softmax(Layer):
    def set_name(self):
        self.name = 'softmax'

    def activation_function(self, Z):
        #prevent overflows by subtracting max values and clipping
        Z = np.clip(Z - np.max(Z), -1e6, 1e6)
        Z_exp = np.exp(Z)
        y_pred = Z_exp / np.sum(Z_exp, axis=-1, keepdims=True)
        return y_pred
    
    def backpropagation(self, y_train):
        delta = y_train * self.derivative_activation_function(np.array(self.output, copy=True))

        # compute the gradients of the loss with respect to the weights and biases
        self.weights_grad = np.dot(self.input.T, delta)
        self.bias_grad = np.sum(delta, axis=0)

        # compute the error of the current layer
        res_delta = np.dot(delta, self.weights.T)
        return res_delta
    
    def derivative_activation_function(self, Z):
        return self.activation_function(Z) * (1 - self.activation_function(Z))