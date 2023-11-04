import numpy as np

class BaseLayer:
    def __init__(self, input, output):
        self.nodes = nodes
        self.input = input
        self.output = output

    def forward(self, input):
        pass

    def backward(self, output_gradient, learning_rate):
        pass
 