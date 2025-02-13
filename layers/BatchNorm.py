from layers.Layer import Layer
import numpy as np

class BatchNorm(Layer):
    def set_name(self):
        self.name = 'batchnorm'

    def backpropagation(self, above_delta):
        x_hat = self.x_hat
        gamma = self.gamma
        N = self.N
        sqrtvar = self.sqrtvar

        self.dgamma = np.sum(above_delta * x_hat, axis=0)
        self.dbeta = np.sum(above_delta, axis=0)

        dx_hat = above_delta * gamma

        sum_dx_hat = np.sum(dx_hat, axis=0)
        sum_dx_hat_x_hat = np.sum(dx_hat * x_hat, axis=0)

        dx = (dx_hat - sum_dx_hat / N - x_hat * sum_dx_hat_x_hat / N) / sqrtvar

        return dx
    
    def feedforward(self, input, train):
        if not hasattr(self, 'gamma'):
            n_features = input.shape[1]
            self.gamma = np.ones(n_features, dtype=np.float32)
            self.beta = np.zeros(n_features, dtype=np.float32)
            self.running_mean = np.zeros(n_features, dtype=np.float32)
            self.running_var = np.ones(n_features, dtype=np.float32)
            self.eps = 1e-5
            self.momentum = 0.9

        if train:
            mean = np.mean(input, axis=0)
            var = np.var(input, axis=0)
            self.x_hat = (input - mean) / np.sqrt(var + self.eps)
            self.N = input.shape[0]
            self.mean = mean
            self.var = var
            self.sqrtvar = np.sqrt(var + self.eps)
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            self.x_hat = (input - self.running_mean) / np.sqrt(self.running_var + self.eps)
        output = self.gamma * self.x_hat + self.beta
        return output

    def activation_function(self, Z):
        return Z
    
    def derivative_activation_function(self, Z):
        return 1