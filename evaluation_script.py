# A note to whoever reads this:
# I realize this is pretty much a mess. Below you will find blocks of code that are commented out, and some that are
# not. I was using the same script to run similar experiments

import os
import numpy as np
import tensorflow as tf
from Model import myModel
from Test import test_model, plot_preds, plot_trends


save_dir = 'saved_models'
########################################################################################################################
# load test datasets and instantiate .Dataset objects to hold them
#nXtest = np.load('cad_90s_nXtest.npy')
#nytest = np.load('cad_90s_nytest.npy')
#
#means = np.load('cad_90s_means_test.npy')
#stds = np.load('cad_90s_stds_test.npy')

nXtest = np.load('cad_90s_nXtest_60day.npy')
nytest = np.load('cad_90s_nytest_60day.npy')
means = np.load('cad_90s_means_test_60day.npy')
stds = np.load('cad_90s_stds_test_60day.npy')

# make into a tf.data.Dataset object
test_ds = tf.data.Dataset.from_tensor_slices((nXtest, nytest)).batch(32)
########################################################################################################################
# load models
cnn_lstm_model = myModel()
lstm_model = myModel()

filename_cnn_lstm = os.path.join(save_dir, '3.32-3.16-3.8_60_day.h5')
filename_lstm = os.path.join(save_dir, 'lstm_60_day.h5')

cnn_lstm_model.load_model(filename_cnn_lstm)
lstm_model.load_model(filename_lstm)

cnn_lstm_result = test_model(cnn_lstm_model.model, test_ds)
lstm_result = test_model(lstm_model.model, test_ds)
########################################################################################################################
# print test loss figures
print("Test loss for CNN-LSTM model is: ")
print(cnn_lstm_result)
print("Test loss for LSTM model is: ")
print(lstm_result)
########################################################################################################################
# generate next-step-ahead prediction plots
denorm_y_test = nytest.flatten() * stds + means


preds = cnn_lstm_model.model.predict(nXtest).flatten()
preds = preds * stds + means

plot_preds(preds, denorm_y_test, "Predictions vs Actual price", "CNN-LSTM")

preds = lstm_model.model.predict(nXtest).flatten()
preds = preds * stds + means

plot_preds(preds, denorm_y_test, "Predictions vs Actual price", "LSTM")

########################################################################################################################
# generate trend prediction plots
#raw_prices = np.load("cad_90s_test_raw.npy")
#forecast_length = 14
#plot_trends(cnn_lstm_model, raw_prices, forecast_length, "CNN-LSTM 14 day forecast")
#plot_trends(lstm_model, raw_prices, forecast_length, "LSTM 14 day forecast")

#raw_prices = np.load("cad_90s_test_raw.npy")
#forecast_length = 60
#plot_trends(cnn_lstm_model, raw_prices, forecast_length, "CNN-LSTM 60 day forecast", hist_length=60)
#plot_trends(lstm_model, raw_prices, forecast_length, "LSTM 60 day forecast", hist_length=60)