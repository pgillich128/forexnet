# Peter Gillich
# model class
# Code adapted from Jakob Aungiers' project, Altum Intelligence Ltd
# >> https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction/blob/master/core/model.py

import os
import math

import keras
import numpy as np
import datetime as dt
from numpy import newaxis
from keras.layers import Dense, Activation, Dropout, LSTM, convolutional, Input
from keras.models import Sequential, load_model
from keras.callbacks import EarlyStopping, ModelCheckpoint
import Train


class myModel:

    def __init__(self):
        self.model = Sequential()

    def load_model(self, filepath):
        print('[Model] Loading model from file %s' % filepath)
        self.model = load_model(filepath)

    def build_model(self, configs):

        self.model.add(keras.Input((configs['problem_spec']['window'],
                                    configs['problem_spec']['features'])))

        for layer in configs['model']['layers']:
            units = layer['units'] if 'units' in layer else None
            dropout_rate = layer['dropout_rate'] if 'dropout_rate' in layer else None
            activation = layer['activation'] if 'activation' in layer else None
            return_seq = layer['return_sequences'] if 'return_sequences' in layer else None
            filters = layer['filters'] if 'filters' in layer else None
            kernel_size = layer['kernel_size'] if 'kernel_size' in layer else None
            padding = layer['padding'] if 'padding' in layer else None
            strides = layer['strides'] if 'strides' in layer else None

            if layer['type'] == 'dense':
                self.model.add(Dense(units, activation=activation))
            if layer['type'] == 'lstm':
                self.model.add(LSTM(units, return_sequences=return_seq))
            if layer['type'] == 'conv1d':
                self.model.add(convolutional.Conv1D(filters=filters,
                                                    kernel_size=kernel_size,
                                                    padding=padding,
                                                    strides=strides))
            if layer['type'] == 'dropout':
                self.model.add(Dropout(dropout_rate))

        self.model.compile(loss=configs['model']['loss'],
                           optimizer=configs['model']['optimizer'],
                           metrics=configs['model']['metric'])
        self.model.build()
        self.model.summary()
        print('[Model] Model Compiled')

    def train(self, trainset, valset, epochs, save_dir, model_spec, batch_size):
        Train.train(self, trainset, valset, epochs, save_dir, model_spec, batch_size)

    def one_day_forecast(self, x):
        """
        :param x: lookback period
        :return pred: forecasted price
        """
        # normalize the input
        mean = np.mean(x)
        std = np.std(x)
        x = (x - mean) / std

        # get prediction
        pred = self.model.predict(x)
        # denormalize
        pred = std * pred + mean

        return pred

    def n_day_forecast(self, x, num_days, history_length=60):
        """
        :param num_days: number of days to forecast ahead
        :param x: lookback period
        :return forecast: numpy array of forecasted prices
        """
        forecast = np.array([])
        for i in range(num_days):
            pred = self.one_day_forecast(x)
            forecast = np.append(forecast, pred)
            x = np.append(x, pred)
            x = x[1:]
            x = x.reshape((1, history_length))

        forecast = np.array(forecast)

        return forecast


