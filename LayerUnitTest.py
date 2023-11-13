import unittest
from Layer import Layer
import numpy as np

class TestLayerMethods(unittest.TestCase):
    def sum_activation(Z):
        np.sum(Z)
    
    def test_computeNode(self):
        Layer test(2, 2, sum_activation)

        test.computeNode(0, [])
        self.assertEqual()
    
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

 