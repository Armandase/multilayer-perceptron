import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

def plot_curve(historic):
    epochs = np.arange(0, historic["epoch"] + 1)
    accu = historic["accu"]
    val_accu = historic["val_accu"]
    loss_entropy = historic["loss_entropy"]
    val_loss_entropy = historic["val_loss_entropy"]
    loss_mse = historic["loss_mse"]
    val_loss_mse = historic["val_loss_mse"]

    # Create the first subplot for the loss at the top
    plt.subplot(3, 1, 1)  # 1 rows, 1 column, first subplot
    plt.plot(epochs, loss_entropy, label='Loss entropy')
    plt.plot(epochs, val_loss_entropy, label='Validation loss entropy')

    plt.title('Loss entropy over epoch')
    plt.xlabel('epoch')
    plt.ylabel('entropy')
    plt.legend()

    # Create the second subplot for the mse a the middle
    plt.subplot(3, 1, 2)  # 2 rows, 1 column, second subplot
    plt.plot(epochs, loss_mse, label='Loss mse')
    plt.plot(epochs, val_loss_mse, label='Validation loss mse')

    plt.title('Loss mse over epoch')
    plt.xlabel('epoch')
    plt.ylabel('mse')
    plt.legend()

    # Create the third subplot for the accuracy at the bottom
    plt.subplot(3, 1, 3)  # 3    rows, 1 column, third subplot
    plt.plot(epochs, accu, label='Accuracy')
    plt.plot(epochs, val_accu, label='Validation accuracy')

    plt.title('Accuracy over epoch')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend()

    # Adjust layout to prevent clipping of titles
    plt.tight_layout()

    # Show the plot
    plt.show()