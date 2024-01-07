from Layer import Layer
import json

weight_path='save_model/model.json'

class Network:
    def __init__(self, learning_rate):
        self.layers = list()
        self.learning_rate = learning_rate

    def addLayers(self, layer):
        if isinstance(layer, Layer) == False:
            print("addLayers: Wrong type of layer")
        
        if len(self.layers) > 0 and layer.weights.shape[0] != self.layers[-1].weights.shape[1]:
            print("addLayers: input len of new layers doesn't match (fully connected neural network)")

        layer.learning_rate = self.learning_rate
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

    def save_weights(self, batch_size=50, epoch=833):
        data = {
            'learning_rate': self.learning_rate,
            'batch_size': batch_size,
            'epoch': epoch,
            'network': []}
        for layer in self.layers:
            layer_data = {
                'name': layer.name,
                'shape': [layer.weights.shape[0], layer.weights.shape[1]],
                'weights': layer.weights.tolist(),
            }
            data['network'].append(layer_data)

        json_data = json.dumps(data, indent=4)
 
        try:
            with open(weight_path, 'w', newline='') as outfile:
                outfile.write(json_data)
            print('Written to file successfully')
        except Exception as e:
            print("An error occurred:", str(e))