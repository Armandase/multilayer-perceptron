from Layer import Layer
import numpy as np

class Relu(Layer):
    def set_name(self):
        self.name = 'relu'

    def backpropagation(self, above_delta):
        above_delta *= self.derivative_activation_function(self.output)

        self.weights_grad = np.dot(self.input.T, above_delta)
        self.bias_grad = np.sum(above_delta, axis=0)

        res_delta = np.dot(above_delta, self.weights.T)
        return res_delta
    
    def activation_function(self, Z):
        return np.maximum(0, Z)
    
    def derivative_activation_function(self, Z):
        return np.where(Z > 0, 1, 0)
    
