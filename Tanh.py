from Layer import Layer
import numpy as np

class Tanh(Layer):
    def set_name(self):
        self.name = 'tanh'

    def backpropagation(self, above_delta):
        above_delta *= self.derivative_activation_function(self.output)

        self.weights_grad = np.dot(self.input.T, above_delta)
        self.bias_grad = np.sum(above_delta, axis=0)

        res_delta = np.dot(above_delta, self.weights.T)
        return res_delta
    
    def activation_function(self, Z):
        return np.tanh(Z)

    # tanh derivative : 1 - tanh^2(Z)
    def derivative_activation_function(self, Z):
        tanh_Z = np.tanh(Z)
        return 1 - np.square(tanh_Z)