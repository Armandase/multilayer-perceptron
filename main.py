import pandas as pd
import numpy as np
import argparse
import yaml
import logging
import os

from parsing import preprocessing
from math_func import derivative_binary_cross_entropy, derivative_subject_binary_cross_entropy, derivate_mean_square_error, derivate_bce
from plotting import plot_curve
from Network import Network
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

def deriv_loss_selection(name):
    if name == "binary_cross_entropy":
        return derivative_binary_cross_entropy
    elif name == "subject_binary_cross_entropy":
        return derivative_subject_binary_cross_entropy
    elif name == "mean_square_error":
        return derivate_mean_square_error
    elif name == "bce":
        return derivate_bce
    else:
        raise Exception("Wrong loss name")

def create_model_from_config(config_model, mean, std):
    cfg_callbacks = config_model['callbacks']

    callbacks = get_callback_from_config(cfg_callbacks)

    loss_func = config_model['loss']
    model = Network(mean, std, callbacks, deriv_loss=deriv_loss_selection(loss_func))

    nb_feature = config_model['input_len']
    lr = config_model['learning_rate']
    
    layers = config_model['layers']
    optimizer = config_model['optimizer']

    above_output = nb_feature
    for layer in layers:
        name = layer['name']
        output = layer['output']
        if name == 'dropout':
            layer = name_to_class(name)(above_output, above_output, learning_rate=lr, dropout_rate=output, optimizer=optimizer)
            output = above_output
        else:
            layer = name_to_class(name)(above_output, output, learning_rate=lr, optimizer=optimizer)
        above_output = output
        model.addLayers(layer)
    model.addLayers(Softmax(output, 2, learning_rate=lr, optimizer=optimizer))
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
        train_x, train_y, test_x, test_y, mean, std = preprocessing(preprocessing_config, verbose)

        config_model = config['model']
        model = create_model_from_config(config_model, mean, std)
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
    try:
        main(args.config)
    except Exception as e:
        print('Error:', e)

