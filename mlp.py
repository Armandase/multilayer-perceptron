import pandas as pd
import numpy as np
import argparse
import yaml
import logging
import os

from Sigmoid import Sigmoid
from Relu import Relu
from Softmax import Softmax
from parsing import preprocessing
from Network import Network
from plotting import plot_curve

def create_model(config_model):
    model = Network()

    nb_feature = config_model['intput_len']
    fc1_output = config_model['fc1']
    fc2_output = config_model['fc2']
    fc3_output = config_model['fc3']
    lr = config_model['learning_rate']

    model.addLayers(Relu(nb_feature, fc1_output, learning_rate=lr))
    model.addLayers(Relu(fc1_output, fc2_output, learning_rate=lr))
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
        data_x, data_y = preprocessing(preprocessing_config, verbose)

    config_model = config['model']
    model = create_model(config_model)

    batch_size = config_model['batch_size']
    epochs = config_model['epochs']
    learning_rate = config_model['learning_rate']
    train_prop = config_model['train_prop']
    test_prop = config_model['test_prop']

    historic = model.fit(data_x, data_y, batch_size, epochs, learning_rate, train_prop, test_prop, verbose)

    plot_curve(historic)
    # net.save_weights()


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', '-c', type=str, default='config.yaml')
    args = argparser.parse_args()
    main(args.config)

