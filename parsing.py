import numpy as np
import pandas as pd
import random

import logging
import os


def one_hot(labels):
    n_values = np.max(labels) + 1
    one_hot_encoding = np.eye(n_values)[labels]

    one_hot_encoding = np.where(one_hot_encoding == 1, 0, 1)
    return one_hot_encoding


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

def get_batches(data_x, data_y, batch_size):
    if len(data_x) != len(data_y):
        logging.error("Data and labels have different lengths")
        exit(1)
    
    batches_x = []
    batches_y = []
    for i in range(0, len(data_x), batch_size):
        batches_x.append(data_x[i:i+batch_size])
        batches_y.append(data_y[i:i + batch_size].astype(int))

    return batches_x, batches_y

def preprocessing(preprocessing_config={}, verbose=False):
    data_path = preprocessing_config['data_path']
    seed = preprocessing_config['seed']
    header = preprocessing_config['header']

    if data_path is None or os.path.exists(data_path) is False:
        logging.error(f"Invalid or missing data file: {data_path}")
        exit(1)

    with open(data_path, 'r') as data_file:
        data = pd.read_csv(data_file, header=None)
        data.columns = header
        if seed != -1:
            logging.info(f"Seed set to {seed}")
            data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
        else:
            logging.info("Seed not set. Randomizing data.")
            data = data.sample(frac=1).reset_index(drop=True)
        data_x = data.iloc[:, 2:].values
        data_y = data.iloc[:, 1].values
        data_y[data_y == 'M'] = 1
        data_y[data_y == 'B'] = 0
    
    if verbose:
        logging.info(f"Data loaded from {data_path}")
        print(data.describe())

    data_x = normalize_data(data_x)
    return data_x, data_y
