import numpy as np
import pandas as pd

import logging
import os


def one_hot(labels):
    labels = labels.astype(int)
    y_one_hot = np.zeros((labels.size, labels.max()+1))
    y_one_hot[np.arange(labels.size), labels] = 1
    return y_one_hot

def split_data(data, train_prop, test_prop): 

    if train_prop + test_prop != 1:
        logging.error("Invalid train and test proportions")
        exit(1)

    train_size = int(len(data) * train_prop)
    test_size = int(len(data) * test_prop)

    # shuffle data
    data = data.sample(frac=1).reset_index(drop=True)

    train = data[:train_size]
    test = data[train_size:train_size + test_size]

    return train, test

def normalize_data(dataset, min_list=None, max_list=None):
    if min_list is None:
        min_list = np.min(dataset, axis=0)
    if max_list is None:
        max_list = np.max(dataset, axis=0)
    dataset = (dataset - min_list) / (max_list - min_list)
    return dataset, min_list, max_list

def normalize_data_mean_std(dataset, mean_list=None, std_list=None):
    if mean_list is None:
        mean_list = np.mean(dataset, axis=0)
    if std_list is None:
        std_list = np.std(dataset, axis=0)
    dataset = (dataset - mean_list) / std_list
    return dataset, mean_list, std_list

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

def shuffle_data(data_x, data_y):
    idx = np.random.permutation(len(data_x))
    return data_x[idx], data_y[idx]

def remove_outliers_from_data(data_x, data_y):
    for i in range(data_x.shape[1]):
        high = np.quantile(data_x[:, i], 0.99)
        low = np.quantile(data_x[:, i] , 0.01)
        
        idx = np.where((data_x[:, i] > low) & (data_x[:, i] < high))
        data_x = data_x[idx]
        data_y = data_y[idx]
    
    return data_x, data_y

def load_dataset(data_path, preprocessing_config, verbose=True, remove_outliers=False):
    header = preprocessing_config['header']
    seed = preprocessing_config['seed']

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

    if remove_outliers is True:
        data_x, data_y = remove_outliers_from_data(data_x, data_y)

    return data_x, data_y

def create_datasets(data_path, preprocessing_config, verbose=True):
    path_save_data = preprocessing_config['path_save_data']
    if verbose:
        logging.info(f"Creating datasets from {data_path}")

    if path_save_data is None or os.path.exists(path_save_data) is False:
        logging.error(f"Invalid path to save data: {path_save_data}")
        exit(1)

    train_prop = preprocessing_config['train_prop']
    test_prop = preprocessing_config['test_prop']

    with open(data_path, 'r') as data_file:
        data = pd.read_csv(data_file, header=None)
    if verbose:
        logging.info(f"Data loaded from {data_path}")
        print(data.describe())

    train, test = split_data(data, train_prop, test_prop)
    train.to_csv(os.path.join(path_save_data, 'data_train.csv'), index=False, header=False)
    test.to_csv(os.path.join(path_save_data, 'data_test.csv'), index=False, header=False)    

def preprocessing(preprocessing_config={}, verbose=True):
    data_path = preprocessing_config['data_path']
    data_train_path = preprocessing_config['data_train_path']
    data_test_path = preprocessing_config['data_test_path']
    path_save_data = preprocessing_config['path_save_data']
    force_creation = preprocessing_config['force_dataset_creation']

    if force_creation == True:
        create_datasets(data_path, preprocessing_config, verbose)
        data_train_path = os.path.join(path_save_data, 'data_train.csv')
        data_test_path = os.path.join(path_save_data, 'data_test.csv')
        print("Force dataset creation")
    elif data_train_path is None or not os.path.exists(data_train_path) \
        or data_test_path is None or not os.path.exists(data_test_path):
        create_datasets(data_path, preprocessing_config, verbose)
        data_train_path = os.path.join(path_save_data, 'data_train.csv')
        data_test_path = os.path.join(path_save_data, 'data_test.csv')
        print("Data not found. Creating datasets")
    train_x, train_y = load_dataset(data_train_path, preprocessing_config, verbose, remove_outliers=False)
    test_x, test_y = load_dataset(data_test_path, preprocessing_config, verbose, remove_outliers=False)

    train_x, mean, std = normalize_data_mean_std(train_x)
    test_x, _, _ = normalize_data_mean_std(test_x, mean, std)
    return train_x, train_y, test_x, test_y, mean, std
