from layers.Layer import Layer
import numpy as np

class Dropout(Layer):
    def __str__(self):
        return self.name + " Layer: input " + str(self.weights.shape[0]) \
            + " output " + str(self.weights.shape[1]) \
            + " dropout rate " + str(self.dropout_rate)
    

    def set_name(self):
        self.name = 'dropout'
        self.mask = None

    def backpropagation(self, above_delta):
        return above_delta

    def feedforward(self, input, train):
        if train:
            input = input * (1 - self.dropout_rate)
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=input.shape) / (1 - self.dropout_rate)
            self.output = input * self.mask
        else:
            self.output = input
        return self.output

    def activation_function(self, Z):
        return Z
    
    def derivative_activation_function(self, Z):
        return 1
    
