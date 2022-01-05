from tensorflow import keras
import os
import pydot
from keras.utils.vis_utils import plot_model

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'
save_dir = 'saved_models'

filename_cnn_lstm = os.path.join(save_dir, '3.32-3.16-3.8.h5')
filename_lstm = os.path.join(save_dir, 'lstm.h5')

model_lstm = keras.models.load_model(filename_lstm)
model_cnn_lstm = keras.models.load_model(filename_cnn_lstm)


lstm_img = 'lstm.jpg'
cnn_lstm_img = 'cnn_lstm.jpg'

keras.utils.plot_model(model_lstm,
                       lstm_img,
                       show_shapes=True,
                       show_dtype=False,
                       dpi = 1000,
                       layer_range=None)

#keras.utils.plot_model(model_cnn_lstm, cnn_lstm_img)


