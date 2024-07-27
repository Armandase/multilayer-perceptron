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

def split_data(data_x, data_y, train_prop, test_prop):
    if len(data_x) != len(data_y):
        logging.error("Data and labels have different lengths")
        exit(1)
    elif train_prop + test_prop != 1:
        logging.error("Invalid train and test proportions")
        exit(1)

    train_size = int(len(data_x) * train_prop)
    valid_size = int(len(data_x) * test_prop)

    # shuffle data
    indices = np.arange(len(data_x))
    np.random.shuffle(indices)
    data_x = data_x[indices]
    data_y = data_y[indices]

    train_x = data_x[:train_size]
    train_y = data_y[:train_size]
    valid_x = data_x[train_size:train_size + valid_size]
    valid_y = data_y[train_size:train_size + valid_size]

    return train_x, train_y, valid_x, valid_y

def normalize_data(dataset):
    min_list = np.min(dataset, axis=0)
    max_list = np.max(dataset, axis=0)
    dataset = (dataset - min_list) / (max_list - min_list)
    return dataset


def get_batches(data_x, data_y, batch_size):
    if len(data_x) != len(data_y):
        logging.error("Data and labels have different lengths")
        exit(1)
    elif batch_size <= 0 or batch_size > len(data_x):
        logging.error("Invalid batch size")
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
