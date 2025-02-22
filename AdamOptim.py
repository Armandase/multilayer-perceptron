import numpy as np

class AdamOptim():
    def __init__(self, eta=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.m_delta_weigths, self.v_delta_weights = 0, 0
        self.m_delta_bias, self.v_delta_bias = 0, 0
        
        self.eta = eta
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = epsilon

        self.name = "adam"

    def update(self, t, w, b, delta_w, delta_b):
        t = t + 1
        # momentum beta1 (moving average)
        self.m_delta_weigths = self.beta1 * self.m_delta_weigths + (1 - self.beta1) * delta_w
        self.m_delta_bias = self.beta1 * self.m_delta_bias + (1 - self.beta1) * delta_b

        #momentum beta2 (moving averaga)
        self.v_delta_weights = self.beta2 * self.v_delta_weights + (1 - self.beta2) * (delta_w ** 2)
        self.v_delta_bias = self.beta2 * self.v_delta_bias + (1 - self.beta2) * (delta_b ** 2)

        # bias correction of the moving averages
        m_delta_weights_corr = self.m_delta_weigths / (1 - self.beta1 ** t)        
        m_delta_bias_corr = self.m_delta_bias / (1 - self.beta1 ** t)        
        v_delta_weights_corr = self.v_delta_weights / (1 - self.beta2 ** t)        
        v_delta_bias_corr = self.v_delta_bias / (1 - self.beta2 ** t)

        # update model
        w = w - self.eta * (m_delta_weights_corr / (np.sqrt(v_delta_weights_corr) + self.eps) )       
        b = b - self.eta * (m_delta_bias_corr / (np.sqrt(v_delta_bias_corr) + self.eps))
        return w, b