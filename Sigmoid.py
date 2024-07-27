from Layer import Layer
import numpy as np

class Sigmoid(Layer):
    def set_name(self):
        self.name = 'sigmoid'

    def backpropagation(self, above_delta):
        above_delta *= self.derivative_activation_function(self.output)

        self.weights_grad = np.dot(self.input.T, above_delta)
        self.bias_grad = np.sum(above_delta, axis=0)

        res_delta = np.dot(above_delta, self.weights.T)
        return res_delta
    
    def activation_function(self, Z):
        return 1 / (1 + np.exp(-Z))
    
    def derivative_activation_function(self, Z):
        return self.activation_function(Z) * (1 - self.activation_function(Z))