import numpy as np

class Layer:
    def __init__(self, nodes, input_len ,funcActivation):
        self.nodes = nodes
        self.bias = np.random.rand(nodes, 1)
        self.activation = funcActivation
        self.weights = np.random.rand(nodes, input_len)

    def computeNode(self, index, input):
        sum = 0
        for i in range(self.weights[index].shape[0]):
            sum += input[i] * self.weights[index][i]
        sum + self.bias
        return (self.activation(sum))
    
    def computeLayer(self, input):
        output = np.array([self.computeNode(i, input) for i in range(self.nodes)])
        print("LOOP:", output[0])
        
        Z1 = self.weights.dot(input) + self.bias
        output = self.activation(Z1)
        print("DOT PRODUCT", output[0])
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

 