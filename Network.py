from Layer import Layer
import yaml

weight_path='save_model/model.yaml'

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

    def save_weights(self):
        data = {'network': []}
        for layer in self.layers:
            layer_data = {
                'layer': layer.name,
                'shape': [layer.weights.shape[0], layer.weights.shape[1]],
                'weights': layer.weights.tolist(),
            }
            data['network'].append(layer_data)
        
        with open(weight_path, 'w', newline='') as file:
            yaml.dump(data,file,sort_keys=False) 
        print('Written to file successfully')