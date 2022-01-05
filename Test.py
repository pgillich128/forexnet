import matplotlib.pyplot as plt
import numpy as np
from Model import myModel

def test_model(model, testset):
    result = model.evaluate(testset)
    return result


def plot_preds(preds, labels, plot_title, model_type):
    """
        A function which plots predicted
        values (from the test set), along with their labels.

        the predictions get de-normalized before plotting,
        hence the other two input arguments
     """
    fig, ax = plt.subplots()
    ax.set_title(plot_title)
    ax.set_ylabel("Price")
    n = len(preds)
    x = np.arange(n)

    ax.plot(x, labels, label="True price")
    ax.plot(x, preds, label=model_type)

    ax.legend()

    plt.savefig(model_type + '60day.pdf')



def plot_trends(model, test_set, forecast_length, title, hist_length=60 ):
    """A function to visualize trend prediction
        test_set should be a numpy array
    """

    fig, ax = plt.subplots()

    ax.set_title(title)
    ax.set_ylabel("Price")
    ax.set_xlabel("Day")

    n = len(test_set)
    t = np.arange(n)

    ax.plot(t, test_set, label="True Price")

    start_hist_index = 0
    start_forecast_index = hist_length + start_hist_index

    while (start_forecast_index + forecast_length -1 < n):
        hist = test_set[start_hist_index : start_hist_index + hist_length]
        hist = hist.reshape((1, hist_length))
        preds = model.n_day_forecast(hist, forecast_length)

        x = np.arange(forecast_length) + start_forecast_index
        ax.plot(x, preds)
        start_hist_index = start_hist_index + forecast_length
        start_forecast_index = start_forecast_index + forecast_length


    ax.legend()

    plt.savefig(title + '60day.pdf')
