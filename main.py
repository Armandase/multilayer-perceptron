import pandas as pd
import numpy as np
import argparse
import yaml
import logging
import os

from Sigmoid import Sigmoid
from Relu import Relu
from Tanh import Tanh
from Softmax import Softmax
from parsing import preprocessing
from Network import Network
from plotting import plot_curve
from Dropout import Dropout
from Callback import Callback

def get_callback_from_config(config_callback):
    enable_early_stop = config_callback ['enable_early_stop']
    early_stop = config_callback['early_stop']
    enable_save_best_model = config_callback ['enable_save_best_model']
    best_model_path = config_callback ['best_model_path']
    callback = Callback(enable_early_stop=enable_early_stop, early_stop_delta=early_stop,
                        enable_save_best_vloss=enable_save_best_model, path_best_model=best_model_path)
    return callback


def create_model_big(config_model):
    callbacks = get_callback_from_config(config_model['callbacks'])

    model = Network(callbacks)

    nb_feature = config_model['intput_len']
    fc1_output = config_model['fc1']
    fc2_output = config_model['fc2']
    fc3_output = config_model['fc3']
    fc4_output = config_model['fc4']
    fc5_output = config_model['fc5']
    lr = config_model['learning_rate']

    model.addLayers(Relu(nb_feature, fc1_output, learning_rate=lr))
    model.addLayers(Dropout(fc1_output, dropout_rate=0.2))
    model.addLayers(Relu(fc1_output, fc2_output, learning_rate=lr))
    model.addLayers(Dropout(fc2_output, dropout_rate=0.2))
    model.addLayers(Relu(fc2_output, fc3_output, learning_rate=lr))
    model.addLayers(Dropout(fc3_output, dropout_rate=0.2))
    model.addLayers(Relu(fc3_output, fc4_output, learning_rate=lr))
    model.addLayers(Dropout(fc3_output, dropout_rate=0.2))
    model.addLayers(Tanh(fc4_output, fc5_output, learning_rate=lr))
    model.addLayers(Softmax(fc5_output, 2, learning_rate=lr))

    return model

def create_model_tiny(config_model):
    callbacks = get_callback_from_config(config_model['callbacks'])

    model = Network(callbacks)

    nb_feature = config_model['intput_len']
    fc1_output = config_model['fc1']
    fc2_output = config_model['fc2']
    fc3_output = config_model['fc3']
    dropout_rate = config_model['dropout_rate']
    lr = config_model['learning_rate']

    if dropout_rate > 1 or dropout_rate < 0:
        raise Exception("Dropout rate should be between 0 and 1.")

    model.addLayers(Relu(nb_feature, fc1_output, learning_rate=lr))
    if dropout_rate > 0:
        model.addLayers(Dropout(fc1_output, dropout_rate=dropout_rate))
    model.addLayers(Relu(fc1_output, fc2_output, learning_rate=lr))
    if dropout_rate > 0:
        model.addLayers(Dropout(fc2_output, dropout_rate=dropout_rate))
    model.addLayers(Relu(fc2_output, fc3_output, learning_rate=lr))
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
    model = create_model_tiny(config_model)

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

