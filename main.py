import pandas as pd
import numpy as np
import argparse
import yaml
import logging
import os

from parsing import preprocessing
from Network import Network
from plotting import plot_curve
from layers.Dropout import Dropout
from Callback import Callback
from layers.Sigmoid import Sigmoid
from layers.Relu import Relu
from layers.Tanh import Tanh
from layers.Softmax import Softmax
from layers.BatchNorm import BatchNorm
from layers.L1Norm import L1Normalization

def get_callback_from_config(config_callback):
    enable_early_stop = config_callback ['enable_early_stop']
    early_stop = config_callback['early_stop']
    enable_save_best_model = config_callback ['enable_save_best_model']
    best_model_path = config_callback ['best_model_path']
    callback = Callback(enable_early_stop=enable_early_stop, early_stop_delta=early_stop,
                        enable_save_best_vloss=enable_save_best_model, path_best_model=best_model_path)
    return callback

def name_to_class(name):
    name = name.lower()
    if name == 'sigmoid':
        return Sigmoid
    elif name == 'relu':
        return Relu
    elif name == 'tanh':
        return Tanh
    elif name == 'softmax':
        return Softmax
    elif name == 'dropout':
        return Dropout
    elif name == 'batchnorm':
        return BatchNorm
    elif name == 'l1_normalization':
        return L1Normalization
    else:
        raise Exception("Invalid layer name")

def create_model_from_config(config_model):
    cfg_callbacks = config_model['callbacks']
    callbacks = get_callback_from_config(cfg_callbacks)

    model = Network(callbacks)

    nb_feature = config_model['input_len']
    lr = config_model['learning_rate']
    
    layers = config_model['layers']
    optimizer = config_model['optimizer']

    above_output = nb_feature
    for layer in layers:
        name = layer['name']
        output = layer['output']
        if name == 'dropout':
            layer = name_to_class(name)(above_output, above_output, learning_rate=lr, dropout_rate=output)
            output = above_output
        else:
            layer = name_to_class(name)(above_output, output, learning_rate=lr)
        above_output = output
        model.addLayers(layer)
    model.addLayers(Softmax(output, 2, learning_rate=lr))
    return model

def create_model_tiny(config_model):
    callbacks = get_callback_from_config(config_model['callbacks'])

    model = Network(callbacks)

    nb_feature = config_model['input_len']
    fc1_output = 64
    fc2_output = 32
    fc3_output = 32
    dropout_rate = config_model['dropout_rate']
    lr = config_model['learning_rate']

    if dropout_rate > 1 or dropout_rate < 0:
        raise Exception("Dropout rate should be between 0 and 1.")

    model.addLayers(Sigmoid(nb_feature, fc1_output, learning_rate=lr))
    # model.addLayers(BatchNorm(fc1_output, fc1_output, learning_rate=lr))
    if dropout_rate > 0:
        model.addLayers(Dropout(fc1_output, dropout_rate=dropout_rate))
    model.addLayers(Sigmoid(fc1_output, fc2_output, learning_rate=lr))
    # model.addLayers(BatchNorm(fc2_output, fc2_output, learning_rate=lr))
    if dropout_rate > 0:
        model.addLayers(Dropout(fc2_output, dropout_rate=dropout_rate))
    model.addLayers(Sigmoid(fc2_output, fc3_output, learning_rate=lr))
    model.addLayers(Softmax(fc3_output, 2, learning_rate=lr))
    return model

def main(config_path: str):
    if config_path is None or os.path.exists(config_path) is False:
        logging.error(f"Invalid or missing config file: {config_path}")
        return
    
    verbose = False
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
        logging.info(f"Config loaded from {config_path}")

        verbose = config['verbose']
        preprocessing_config = config['preprocessing']
        train_x, train_y, test_x, test_y = preprocessing(preprocessing_config, verbose)

    config_model = config['model']
    # model = create_model_tiny(config_model)
    # print("Tiny model:", model)
    model = create_model_from_config(config_model)
    # print("From config:", model)
    # print("Start training")
    # exit()
    batch_size = config_model['batch_size']
    epochs = config_model['epochs']
    learning_rate = config_model['learning_rate']

    historic = model.fit(train_x, train_y, test_x, test_y, batch_size, epochs, learning_rate, verbose)

    plot_curve(historic)
    print("Training completed")
    model.save_weights(config_model['model_path'])


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', '-c', type=str, default='config.yaml')
    args = argparser.parse_args()
    # try:
    main(args.config)
    # except Exception as e:
        # print('Error:', e)

