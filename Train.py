# Model training
# graphing code borrowed from
# >> https://stackoverflow.com/questions/36952763/how-to-return-history-of-validation-loss-in-keras


import os
import datetime as dt

import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping, ModelCheckpoint

def generate_learning_curves(history, num_epochs, fname):
    '''visualize learning curves'''
    fig, ax = plt.subplots()

    ax.set_xlabel("Epoch")

    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    xc = np.arange(np.size(val_loss))

    plt.plot(xc, train_loss, label="Training loss")
    plt.plot(xc, val_loss, label="Validation loss")
    ax.legend()

    fname = fname + '.pdf'

    plt.savefig(fname)



def train(model, trainset, valset, epochs, save_dir, model_spec, batch_size):
    print("Training started")
    print('[Model] %s epochs, %s batch size' % (epochs, batch_size))

    save_fname = os.path.join(save_dir, model_spec)
    #save_fname = os.path.join(save_dir, '%s-e%s.h5' % (dt.datetime.now().strftime('%d%m%Y-%H%M%S'), str(epochs)))
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=50),
        ModelCheckpoint(filepath=save_fname, monitor='val_loss', save_best_only=True, mode='min')
    ]
    history = model.model.fit(
        trainset,
        validation_data=valset,
        epochs=epochs,
        callbacks=callbacks
    )
    model.model.save(save_fname)

    generate_learning_curves(history, num_epochs=epochs, fname=save_fname)

    print('[Model] Training Completed. Model saved as %s' % save_fname)