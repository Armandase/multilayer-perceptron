import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

def plot_curve(epoch, loss, val_loss):
    # loss = np.convolve(loss, np.ones(3), "valid") / 3
    # val_loss = np.convolve(val_loss, np.ones(3), "valid") / 3
    # epoch = np.convolve(epoch, np.ones(3), "valid") / 3

    interpolateur = interp1d(epoch, loss, kind='linear')

    # Générer des points intermédiaires pour une courbe plus lisse
    x_interpol = np.linspace(epoch.min(), epoch.max(), 10)
    loss_interpol = interpolateur(x_interpol)
    val_loss_interpol = interpolateur(x_interpol)

    plt.plot(x_interpol, loss_interpol, label='Loss')
    plt.plot(x_interpol, val_loss_interpol, label='Validation loss')

    plt.title('Loss over epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.show()