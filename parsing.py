import numpy as np
import pandas as pd
import random

from constants import *

def one_hot(labels):
    """
    Converts categorical labels into one-hot encoded representation.

    Parameters:
    - labels (numpy array): 1-dimensional array containing categorical labels.

    Returns:
    - numpy array: 2-dimensional array representing the one-hot encoded labels.

    Example:
    >>> labels = np.array([0, 1, 1, 0])
    >>> one_hot(labels)
    array([[1, 0],
           [0, 1],
           [0, 1],
           [1, 0]])
    """
    one_hot = np.zeros((labels.size, labels.max() + 1))
    n_values = np.max(labels) + 1
    one_hot = np.eye(n_values)[labels]

    one_hot = np.where(one_hot == 1, 0, 1)
    return one_hot

def normalize_data(dataset):
    """
    Normalizes the input dataset by scaling each feature to a range of [0, 1].
    0 represents the minimum value and 1 the maximum.

    Parameters:
    - dataset (numpy.ndarray): The input dataset with numerical values.

    Returns:
    - numpy.ndarray: The normalized dataset with values scaled to the range [0, 1].
    """
    min_list = np.min(dataset, axis=0)
    max_list = np.max(dataset, axis=0)
    dataset = (dataset - min_list) / (max_list - min_list)
    return dataset
    
def init_data(data_x, data_y, batch_size):
    """
    Randomize and normalize training and validation datasets

    Parameters:
    - data_x (pd.DataFrame): The input features dataset.
    - data_y (np.array): The corresponding labels array.
    - batch_size (int): The size of the training batch.

    Returns:
    - x_train : Normalized training input features (len of batch_size).
    - y_train : Training labels corresponding to x_train (len of batch_size).
    - x_valid : Normalized validation input features (len of data_x sub batch_size).
    - y_valid : Validation labels corresponding to x_valid (len of data_x sub batch_size).

    The function randomly selects 'batch_size' samples for training from data_x,
    and uses the remaining samples for validation. It normalizes the input features
    using the mini-max normalization before returning the datasets.
    """

    indexes = np.random.randint(0, data_x.shape[0], batch_size)
    remaining_indexes = np.setdiff1d(np.arange(data_x.shape[0]), indexes)

    y_train = np.array(data_y[indexes])
    x_train = data_x.iloc[indexes].values
    x_train = normalize_data(np.array(x_train))

    y_valid = np.array(data_y[remaining_indexes])

    x_valid = data_x.iloc[remaining_indexes].values
    x_valid = normalize_data(np.array(x_valid))
    return x_train, y_train, x_valid, y_valid

def handle_args(args):
    data_path: str = None
    epochs: float = None
    batch_size: int = None
    learning_rate: float = None

    if args.data_path:
        data_path = args.data_path
    else:
        data_path = DATA_PATH

    if args.epochs:
        epochs = float(args.epochs)
    else:
        epochs = EPOCHS

    if args.batch_size:
        batch_size = int(args.batch_size)
    else:
        batch_size = BATCH_SIZE

    if args.learning_rate:
        learning_rate = float(args.learning_rate)
    else:
        learning_rate = LEARNING_RATE

    return data_path, epochs, batch_size, learning_rate
    