from Layer import Layer
import numpy as np

class Sigmoid(Layer):
    def set_name(self):
        self.name = 'sigmoid'

    def backpropagation(self, above_delta):

        delta = above_delta * self.derivative_activation_function(self.output)

        self.delta = delta

        res = np.dot(delta, self.weights.T)
        return res
    
    def activation_function(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def derivative_activation_function(self, Z):
        return self.activation_function(Z) * (1 - self.activation_function(Z))