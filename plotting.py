import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

def plot_curve(epoch, loss, val_loss):
    # interpolateur = interp1d(epoch, loss, kind='linear')

    # Générer des points intermédiaires pour une courbe plus lisse
    # x_interpol = np.linspace(epoch.min(), epoch.max(), 50)
    # loss_interpol = interpolateur(x_interpol)
    # val_loss_interpol = interpolateur(x_interpol)

    plt.plot(epoch, loss, label='Loss')
    # plt.plot(x_interpol, loss_interpol, label='Loss')
    # plt.plot(x_interpol, val_loss_interpol, label='Validation loss')
    plt.plot(epoch, val_loss, label='Validation loss')

    plt.title('Loss over epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    plt.show()