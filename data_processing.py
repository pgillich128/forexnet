# Peter Gillich
# script for generating windowed, normalized datasets.
# logic for saving the normalized datasets, as well as normalization parameters, for later use.




import pandas as pd
import numpy as np
import tensorflow as tf
from Datagen import datagen

def make_windowed(dataframe, width, stride=1):
    '''
    takes a time series (pandas dataframe), and returns the sequence of 'windows'
    of 'width' size, sliding across the whole time series with stride 'stride'.
    window of raw prices is ==> [p_t, p_t+1, ..., p_t+n-1], label ==> p_t+n

    :param dataframe: pandas dataframe of price data over time.
    :param width: window width
    :return: dataset with batch size 1, to iterate over for normalization purposes later.
    '''

    dataframe = np.array(dataframe)
    labels = dataframe[width:]
    dataset = tf.keras.preprocessing.timeseries_dataset_from_array(dataframe,
                                                                   labels,
                                                                   batch_size=1,
                                                                   sequence_length=width
                                                                   )

    return dataset


def normalize(dataset):
    '''
    this function takes in a time series dataset, consisting of overlapping windows of fixed length.
    It computes the mean and standard deviation of the data in each window, then centers and normalizes the
    data points in the window. The same transformation is applied to the label (the following day's price)

    :param X: a tf.Dataset with entries formatted as (time series, label). batch size must be equal to 1.
    :return nX, ny, means, stds: tuple, normalized X, normalized y, window-wise means and standard deviations
    '''
    normalized_X = []
    normalized_y = []
    means = []
    stds = []

    for x, y in dataset:
        xnorm = np.array(x[0])
        ynorm = np.array(y)

        mean = np.mean(xnorm)
        std = np.std(xnorm)

        xnorm = (xnorm - mean) / std
        ynorm = (ynorm - mean) / std

        normalized_X.append(xnorm)
        normalized_y.append(ynorm)
        means.append(mean)
        stds.append(std)

    normalized_X = np.array(normalized_X)
    normalized_y = np.array(normalized_y)
    means = np.array(means)
    stds = np.array(stds)

    return normalized_X, normalized_y, means, stds

#########################################################################################
# global test data
#em = pd.read_csv("EM.csv", header=1)
g10 = pd.read_csv("G10.csv")
g10.fillna(-1, inplace=True)
#em.fillna(-1, inplace=True)

#em.rename(columns={'Unnamed: 0': 'date',
#                  'China, CNY per USD': 'CNY',
#                  'China, CNH per USD': 'CNH',
#                  'South Korea, KRW per USD': 'KRW',
#                  'Taiwan, TWD per USD': 'TWD',
#                  'Brazil, BRL per USD': 'BRL',
#                  'Russia, RUB per USD': 'RUB',
#                  'India, INR per USD': 'INR',
#                  'Mexico, MXN per USD': 'MXN',
#                  'Turkey, TRY per USD': 'TRY',
#                  'South Africa, ZAR per USD': 'ZAR',
#                  }, inplace=True)

#em = em.set_index('date')

g10.rename(columns={'Unnamed: 0': 'date',
                    'Euro Area, FX Spot Rates, Macrobond, EUR per USD': 'EUR',
                    'Japan, FX Spot Rates, Macrobond, JPY per USD': 'JPY',
                    'South Korea, KRW per USD': 'KRW',
                    'United Kingdom, FX Spot Rates, Macrobond, GBP per USD': 'GBP',
                    'Switzerland, FX Spot Rates, Macrobond, CHF per USD': 'CHF',
                    'Canada, FX Spot Rates, Macrobond, CAD per USD': 'CAD',
                    'Australia, FX Spot Rates, Macrobond, AUD per USD': 'AUD',
                    'New Zealand, FX Spot Rates, Macrobond, NZD per USD': 'NZD',
                    'Norway, FX Spot Rates, Macrobond, NOK per USD': 'NOK',
                    'Sweden, FX Spot Rates, Macrobond, SEK per USD': 'SEK',
                    'United States, FX Indices, ICE, U.S. Dollar Index, Close': 'ICE',
                    }, inplace=True)

g10 = g10.set_index('date')

cad = g10.loc[g10['CAD'] > 0]
cad = cad.loc[:, 'CAD']
cad_90s = cad.loc['1990-01-01':]

eur = g10.loc[g10['EUR']>0]
eur = eur.loc[:,'EUR']
##########################################################################################
# synthetic dataset
#n=8000
#eps = 0.0025
#synth = datagen(n, eps)
#
#val_index = int(n * 0.9)
#train_index = int(n * 0.7)
#
#train = synth[0:train_index]
#val = synth[train_index : val_index]
#test = synth[val_index:]
##########################################################################################
# split: 0.7 train, 0.2 validation, 0.1 test
n = len(cad_90s)
train = cad_90s[0:int(n * 0.7)]
val = cad_90s[int(n * 0.7): int(n * 0.9)]
test = cad_90s[int(n * 0.9):]

np.save("cad_90s_test_raw", np.asarray(test))

#n = len(cad)
#train = cad[0:int(n * 0.7)]
#val = cad[int(n * 0.7): int(n * 0.9)]
#test = cad[int(n * 0.9):]

#n = len(eur)
#train = eur[0:int(n * 0.7)]
#val = eur[int(n * 0.7): int(n * 0.9)]
#test = eur[int(n * 0.9):]


##########################################################################################
# normalize the train, validation and test splits.
window_width = 60
train_ds = make_windowed(train, window_width)
val_ds = make_windowed(val, window_width)
test_ds = make_windowed(test, window_width)

nXtrain, nytrain, _, _ = normalize(train_ds)

nXval, nyval, val_mean, val_std = normalize(val_ds)

nXtest, nytest, test_mean, test_std = normalize(test_ds)

#np.save("cad_90s_nXtrain", nXtrain)
#np.save('cad_90s_nytrain', nytrain)
#np.save('cad_90s_nXval', nXval)
#np.save('cad_90s_nyval', nyval)
#np.save('cad_90s_nXtest', nXtest)
#np.save('cad_90s_nytest', nytest)
#np.save('cad_90s_means_test', test_mean)
#np.save('cad_90s_stds_test, test_std)

np.save("cad_90s_nXtrain_60day", nXtrain)
np.save('cad_90s_nytrain_60day', nytrain)
np.save('cad_90s_nXval_60day', nXval)
np.save('cad_90s_nyval_60day', nyval)
np.save('cad_90s_nXtest_60day', nXtest)
np.save('cad_90s_nytest_60day', nytest)
np.save('cad_90s_means_val_60day', val_mean)
np.save('cad_90s_stds_val_60day', val_std)
np.save('cad_90s_means_test_60day', test_mean)
np.save('cad_90s_stds_test_60day', test_std)

#
#np.save("cad_nXtrain", nXtrain)
#np.save('cad_nytrain', nytrain)
#np.save('cad_nXval', nXval)
#np.save('cad_nyval', nyval)
#np.save('cad_nXtest', nXtest)
#np.save('cad_nytest', nytest)
#np.save('cad_means_val', val_mean)
#np.save('cad_stds_val', val_std)
#np.save('cad_means_test', test_mean)
#np.save('cad_stds_test', test_std)

#np.save("eur_nXtrain", nXtrain)
#np.save('eur_nytrain', nytrain)
#np.save('eur_nXval', nXval)
#np.save('eur_nyval', nyval)
#np.save('eur_nXtest', nXtest)
#np.save('eur_nytest', nytest)
#np.save('eur_means_val', val_mean)
#np.save('eur_stds_val', val_std)
#np.save('eur_means_test', test_mean)
#np.save('eur_stds_test', test_std)

#np.save("syn_nXtrain", nXtrain)
#np.save('syn_nytrain', nytrain)
#np.save('syn_nXval', nXval)
#np.save('syn_nyval', nyval)
#np.save('syn_nXtest', nXtest)
#np.save('syn_nytest', nytest)
#np.save('syn_means_val', val_mean)
#np.save('syn_stds_val', val_std)
#np.save('syn_means_test', test_mean)
#np.save('syn_stds_test', test_std)