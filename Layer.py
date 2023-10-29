import numpy as np

class Layer:
    def __init__(self, nodes, funcActivation, bias):
        self.nodes = nodes
        self.bias = bias
        self.activation = funcActivation
        self.weights = np.random.randn(nodes, nodes)

    def computeNode(self, index, input):
        sum = 0
        for i in range(self.nodes):
            sum += input[i] * self.weights[index][i]
        sum + self.bias
        return (self.activation(sum))
    
    def computeLayer(self, input):
        output = []
        for i in range(self.nodes):
            output.append(self.computeNode(i, input))
        return output

    def binaryCrossEntropy(output, y_train):
        result = - 1/y_train.shape[0]
        sum = 0
        mean = 0
        for element in output:
            mean += element
        mean /= len(output)
        for i in range(y_train.shape[0]):
            sum += (y_train[i] * math.log(mean)) + ((1 - y_train[i]) * math.log(1 - mean))
        return result * sum

 