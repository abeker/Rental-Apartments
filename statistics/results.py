import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

def prediction_plot(y_test, y_pred):
    ax1 = sns.distplot(y_test, hist = False, color = 'g', label = 'true')
    sns.distplot(y_pred, hist = False, color = 'r', label = 'predicted', ax = ax1)
    plt.gca().set(title='Results', xlabel='values', ylabel='count')
    plt.show()

def prediction_histogram(y_test, y_pred):
    kwargs = dict(alpha=0.5, bins=70)
    plt.hist(np.array(y_test), **kwargs, color='g', label='true')
    plt.hist(np.array(y_pred), **kwargs, color='r', label='predicted')
    plt.gca().set(title='Results', xlabel='values', ylabel='count')
    plt.legend()
    plt.show()
