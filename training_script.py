# driver code for experiments
# partially borrowed from Jakob Aungiers' (Altum Intelligence ltd) Github repo:
# >> https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction/blob/master/core/model.py
import keras
import numpy as np
import tensorflow as tf
from keras import layers, optimizers, losses
from Model import myModel
import json

WIDTH = 30
BATCH_SIZE = 32
EPOCHS=500
save_dir = 'saved_models'

########################################################################################################################
# Select and load a dataset. This is janky--I apologize

#nXtrain = np.load('cad_90s_nXtrain.npy')
#nytrain = np.load('cad_90s_nytrain.npy')
#nXval = np.load('cad_90s_nXval.npy')
#nyval = np.load('cad_90s_nyval.npy')

nXtrain = np.load('cad_90s_nXtrain_60day.npy')
nytrain = np.load('cad_90s_nytrain_60day.npy')
nXval = np.load('cad_90s_nXval_60day.npy')
nyval = np.load('cad_90s_nyval_60day.npy')


#nXtrain = np.load('cad_nXtrain.npy')
#nytrain = np.load('cad_nytrain.npy')
#nXval = np.load('cad_nXval.npy')
#nyval = np.load('cad_nyval.npy')
#nXtest = np.load('cad_nXtest.npy')
#nytest = np.load('cad_nytest.npy')

#nXtrain = np.load('eur_nXtrain.npy')
#nytrain = np.load('eur_nytrain.npy')
#nXval = np.load('eur_nXval.npy')
#nyval = np.load('eur_nyval.npy')
#nXtest = np.load('eur_nXtest.npy')
#nytest = np.load('eur_nytest.npy')


## synthetic dataset to troubleshoot low validation error
#nXtrain = np.load('syn_nXtrain.npy')
#nytrain = np.load('syn_nytrain.npy')
#nXval = np.load('syn_nXval.npy')
#nyval = np.load('syn_nyval.npy')
#nXtest = np.load('syn_nXtest.npy')
#nytest = np.load('syn_nytest.npy')

########################################################################################################################
# instantiate tensorflow Datasets
train_ds = tf.data.Dataset.from_tensor_slices((nXtrain, nytrain)).batch(BATCH_SIZE).shuffle(20000)
val_ds = tf.data.Dataset.from_tensor_slices((nXval, nyval)).batch(BATCH_SIZE)
########################################################################################################################

# Build model from config file and train the model.

model = myModel()
configs = json.load(open('lstm_60day.json', 'r'))
model_spec = "lstm_60_day"
#model_spec = "lstm"
#configs = json.load(open('lstm.json', 'r'))
model.build_model(configs)
model.train(train_ds, val_ds, EPOCHS, save_dir, model_spec + '.h5', batch_size=BATCH_SIZE)
