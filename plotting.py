import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

def plot_curve(historic):
    epoch = historic[:, 0]
    accu = historic[:, 1]
    val_accu = historic[:, 2]
    loss_entropy = historic[:, 3]
    val_loss_entropy = historic[:, 4]
    loss_mse = historic[:, 5]
    val_loss_mse = historic[:, 6]

    plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
    plt.plot(epoch, loss_entropy, label='Loss entropy')
    plt.plot(epoch, loss_mse, label='Loss mse')
    plt.plot(epoch, val_loss_entropy, label='Validation loss entropy')
    plt.plot(epoch, val_loss_mse, label='Validation loss mse')

    plt.title('Loss over epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend()

    # Create the second subplot for the validation loss
    plt.subplot(2, 1, 2)  # 2 rows, 1 column, second subplot
    plt.plot(epoch, accu, label='Accuracy')
    plt.plot(epoch, val_accu, label='Validation accuracy')

    plt.title('Accuracy over epoch')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()

    # Show the plot
    plt.show()