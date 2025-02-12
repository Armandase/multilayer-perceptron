from layers.Layer import Layer
import numpy as np

class L1Normalization(Layer):
    def set_name(self):
        self.name = 'l1_normalization'

    def backpropagation(self, above_delta):
        x = self.x
        s = self.s
        sum_term = np.sum(above_delta * x, axis=1, keepdims=True)
        sign_x = np.sign(x)
        dx = (above_delta * s - sign_x * sum_term) / (s ** 2)
        return dx

    def feedforward(self, input, train):
        self.eps = 1e-8
        self.x = input
        self.s = np.sum(np.abs(input), axis=1, keepdims=True) + self.eps
        return input / self.s

    def activation_function(self, Z):
        return Z
    
    def derivative_activation_function(self, Z):
        return 1