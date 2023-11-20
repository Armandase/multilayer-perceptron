import numpy as np

class Layer:
    def __init__(self, nodes, input_len ,funcActivation):
        self.nodes = nodes
        # self.bias = np.random.rand(nodes, 1)
        self.bias = np.zeros(nodes)
        self.activation = funcActivation
        self.weights = np.random.rand(nodes, input_len)
        self.learning_rate = 1

    def computeLayer(self, input):
        
        # Z1 = self.weights.dot(input) + self.bias
        Z1 = self.weights.dot(input)
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
    
    def deltaOutputLayer(self, waited_output, output):
        delta = output * (1 - output) * (waited_output - output)
        return delta
    
    def deltaHiddenLayer(self, above_delta, output, above_weights):
        delta = np.zeros(output.shape[0])
        above_weights = above_weights[0]
        for i in range(output.shape[0]):
            tmp =  output[i] * (1 - output[i])
            delta[i] = tmp * np.sum(above_delta * above_weights[i])
        return delta
    
    def update_weights(self, output, delta):
        delta_weights = self.learning_rate * output.dot(delta)
        self.weights += delta_weights
