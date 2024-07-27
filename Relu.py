from Layer import Layer
import numpy as np

class Relu(Layer):
    def set_name(self):
        self.name = 'relu'

    def backpropagation(self, above_delta):

        delta = above_delta * self.derivative_activation_function(self.output)

        self.delta = delta

        res = np.dot(above_delta, self.weights.T)
        return res
    
    def activation_function(self, Z):
        return np.maximum(0, Z)
    
    def derivative_activation_function(self, Z):
        return np.where(Z > 0, 1, 0)
    
