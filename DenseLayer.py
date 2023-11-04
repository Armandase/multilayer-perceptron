import numpy as np

class DenseLayer:
    def __init__(self, nodes, input_len ,funcActivation):
        self.nodes = nodes
        self.activation = funcActivation
        self.weights = np.random.rand(nodes, input_len)
        self.bias = np.random.rand(nodes, 1)
    
    def forward(self, input):
        # store input for backward prop too
        self.input = input
        # Z is the matrix sent to the activation function
        # it's computed by the dot product of the weights with the inputs, plus bias's matrix
        Z1 = self.weights.dot(self.input) + self.bias
        # the output (Y matrix) is the result of the activation function with Z as its parameter
        output = self.activation(Z1)
        return output
    
    def backward(self, output_gradient, learning_rate):
        weights_gradient = output_gradient.dot(self.input.T)  
        self.weights -= learning_rate * weights_gradient
        self.bias -= learning_rate * output_gradient
        # return input of the layer above
        return self.weights.T.dot(output_gradient)   

  