from Layer import Layer

class Network:
    def __init__(self):
        self.layers = list()

    def addLayers(self, layer):
        if isinstance(layer, Layer) == False:
            print("addLayers: Wrong type of layer")
        
        if len(self.layers) > 0 and layer.weights.shape[0] != self.layers[-1].weights.shape[1]:
            print("addLayers: input len of new layers doesn't match (fully connected neural network)")

        self.layers.append(layer)

    def feedforward(self, input, train):
        X = input

        for layer in self.layers:
            X = layer.feedforward(X, train)
        return X

    def backpropagation(self, y_train):
        gradient = y_train

        for layer in reversed(self.layers):
            gradient = layer.backpropagation(gradient)