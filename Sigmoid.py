from Layer import Layer
import numpy as np

class Sigmoid(Layer):
    def backpropagation(self, above_delta):
        delta = above_delta * self.output * (1 - self.output) 

        delta_weights = np.dot(self.input.T, delta) * self.learning_rate
        self.weights = self.weights - delta_weights

        delta_bias = np.sum(delta, axis=0)
        self.bias = self.bias - delta_bias * self.learning_rate

        res = np.dot(delta, self.weights.T)
        return res
    
    def activation_function(self, Z):
        return (1 / (1 + np.exp(-Z)))