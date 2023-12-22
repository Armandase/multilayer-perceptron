from Layer import Layer

class Network:
    def __init__(self):
        self.layers = list()
        self.inputs = list()
    def addLayers(self, layer):
        if isinstance(layer, Layer) == False:
            print("addLayers: Wrong type of layer")
        
        if len(self.layers) > 0 and layer.input_len != self.layers[-1].nodes:
            print("addLayers: input len of new layers doesn't match (fully connected neural network)")  
        self.layers.append(layer)