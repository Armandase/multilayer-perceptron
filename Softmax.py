from Layer import Layer
import numpy as np
from parsing import *
from math_func import *

class Softmax(Layer):
    def set_name(self):
        self.name = 'softmax'

    def activation_function(self, Z):
        #prevent overflows by subtracting max values
        # Z_exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        # Z = Z - np.max(Z, axis=1, keepdims=True)
        Z = Z - np.max(Z)
        Z = np.clip(Z, -500, 500)
        Z_exp = np.exp(Z)
        y_pred = Z_exp / np.sum(Z_exp, axis=1, keepdims=True)
        return y_pred
    
    def backpropagation(self, y_train):
        error = y_train
        error[np.arange(y_train.shape[0]), y_train.argmax(axis=1).astype(np.int64)] -= 1
        error /= len(y_train)

        # deriv = binary_cross_entropy_derivative(y_train, self.output)
        # delta = error * self.derivative_activation_function(self.output)

        self.delta = error

        delta = error * self.derivative_activation_function(self.output)


        # res = np.dot(self.weights, delta.T)
        res = np.dot(delta, self.weights.T)
        # res *= 
        return res
    
    def derivative_activation_function(self, Z):
        return self.activation_function(Z) * (1 - self.activation_function(Z))