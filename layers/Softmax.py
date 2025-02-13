from layers.Layer import Layer
import numpy as np
from parsing import *
from math_func import *

class Softmax(Layer):
    def set_name(self):
        self.name = 'softmax'

    def activation_function(self, Z):
        #prevent overflows by subtracting max values and clipping
        # Z = np.clip(Z - np.max(Z, axis=1, keepdims=True), 1e-15, -1e-15)
        Z = Z - np.max(Z, axis=1, keepdims=True)
        Z_exp = np.exp(Z)
        y_pred = Z_exp / np.sum(Z_exp, axis=-1, keepdims=True)
        return y_pred
    
    def backpropagation(self, error):
        # error = error * self.derivative_activation_function(self.output)
        error = error / (self.input.shape[0])
        
        self.weights_grad = np.dot(self.input.T, error)
        self.bias_grad = np.sum(error, axis=0)

        res_delta = np.dot(error, self.weights.T)
        return res_delta

    def derivative_activation_function(self, Z):
        return self.activation_function(Z) * (1 - self.activation_function(Z))