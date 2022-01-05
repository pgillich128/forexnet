Hi, and thanks for reading!

Enclosed are saved, trained neural network models (in the saved_models directory), along with some raw data (EM = emerging markets, G10=g10 countries)
saved as .csv files.

The workflow is as follows: first, one must run the data_processing script to generate train, test, and validation sets, as well as 
normalizing these for eventually training neural networks. we used z-score normalization (see the report), and for the purposes
of visualizing predictions against the ground truth, had to save the means and variances to denormalize, so don't forget that.

the main dataset is the 'cad_90s' series, whose train/validation/test splits are saved as .npy files for use in the training_script
as well as in the evaluation_script--therefore, only run the data_processing script once, as the other scripts use these to train and
test. There is also the ability to create a synthetic dataset--simply uncomment the applicable block of code from data_processing.

The training_script needs a .json config file (idea credit: Jakob Aungiers, Altum Intelligence Ltd, github:https://github.com/jaungiers/LSTM-Neural-Network-for-Time-Series-Prediction)
to define the layers of a given neural network--just identify/modify the config file you want (or write your own), update the 
file name which is passed as argument to myModel.build_model() in order to build and train. 

In the Test file, there are methods to generate plots. I did a bad thing and hard-coded some parameters--in particular, the 
price history length is defaulted to 60 days. If you want to change that, you'll have to go through the plotting methods to change the
default window_width (history_length in some places). There is also a hard-coded default window length of 60 in the Model class,
in the n_day_forecast method.

I hope this code does not give you too much trouble--I know it could be much cleaner, but I've attempted to explain it as best
I can at a high level here, and I believe it's not too hard to figure out if you have a broad idea of what I was doing.






