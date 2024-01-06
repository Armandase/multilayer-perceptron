from Layer import Layer
import numpy as np
from parsing import *

class Softmax(Layer):
    def set_name(self):
        self.name = 'softmax'

    def backpropagation(self, y_train):
        one_hot_y_train = one_hot(y_train)
        delta = self.output - one_hot_y_train

        delta_weights = np.dot(self.input.T, delta) * self.learning_rate
        self.weights = self.weights - delta_weights

        delta_bias = np.sum(delta, axis=0)
        self.bias = self.bias - delta_bias * self.learning_rate

        res = np.dot(delta, self.weights.T)
        return res

    def activation_function(self, Z):
        #prevent overflows by subtracting max values
        Z_exp = np.exp(Z - np.max(Z, axis=1, keepdims=True))
        y_pred = Z_exp / np.sum(Z_exp, axis=1, keepdims=True)
        return y_pred