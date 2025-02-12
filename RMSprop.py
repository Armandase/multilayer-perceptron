import numpy as np

class RMSprop():
    def __init__(self, eta=0.01, epsilon=1e-8, alpha=0.9):
        self.name = "rmsprop"
        self.eps = epsilon
        self.lr = eta
        self.alpha = alpha
        self.v_delta_weights = 0
        self.v_delta_bias = 0
        
    def update(self, t, w, b, delta_w, delta_b):
        self.v_delta_weights = self.alpha * self.v_delta_weights + (1 - self.alpha) * (delta_w ** 2)
        self.v_delta_bias = self.alpha * self.v_delta_bias + (1 - self.alpha) * (delta_b ** 2)
        
        w = w - (self.lr / np.sqrt(self.v_delta_weights + self.eps)) * delta_w     
        b = b - (self.lr / np.sqrt(self.v_delta_bias + self.eps)) * delta_b
        
        return w, b