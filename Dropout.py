from Layer import Layer
import numpy as np

class Dropout(Layer):
    def set_name(self):
        self.name = 'dropout'
        self.mask = None

    def backpropagation(self, above_delta):
        return above_delta

    def feedforward(self, input, train):
        if train:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=input.shape) / (1 - self.dropout_rate)
            self.output = input * self.mask
        else:
            self.output = input
        return self.output


    def activation_function(self, Z):
        return Z
    
    def derivative_activation_function(self, Z):
        return 1
    
