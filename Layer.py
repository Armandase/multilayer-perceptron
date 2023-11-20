import numpy as np

class Layer:
    def __init__(self, nodes, input_len ,funcActivation):
        self.nodes = nodes
        self.bias = np.random.rand(nodes, 1)
        self.activation = funcActivation
        self.weights = np.random.rand(nodes, input_len)

    def computeLayer(self, input):
        Z1 = self.weights.dot(input) + self.bias
        output = self.activation(Z1)
        return output

    def binaryCrossEntropy(output, y_train):
        result = - 1 / y_train.shape[0]
        sum = 0
        mean = 0
        for element in output:
            mean += element
        mean /= len(output)
        for i in range(y_train.shape[0]):
            sum += (y_train[i] * math.log(mean)) + ((1 - y_train[i]) * math.log(1 - mean))
        return result * sum
    
    def cost(output, waited): 
        return (output - waited) ** 2
        
    def sigmoid(Z):
        return (1 / (1 + np.exp(-Z)))
    def softmax(Z):
        expZ = np.exp(Z)
        return expZ / np.sum(expZ)

    
