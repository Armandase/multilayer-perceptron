import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d

def plot_curve(epoch, loss, val_loss, accu, val_accu):
    plt.subplot(2, 1, 1)  # 2 rows, 1 column, first subplot
    plt.plot(epoch, loss, label='Loss')
    plt.plot(epoch, val_loss, label='Validation loss')

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