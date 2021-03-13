from sklearn.datasets import load_boston
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

def boxplot(dataframe, column):
    print('---------- outliers ----------')
    print(dataframe.sort_values(by=[column], ascending=False).head(100).to_string())
    sns.set_theme(style="whitegrid")
    sns.boxplot(x=dataframe[column])
    plt.show()
    print('---------- outliers end ----------')

def subplots(dataframe):
    df = dataframe.drop('id', axis=1)
    fig, ax = plt.subplots(figsize=(16, 8))
    ax = sns.boxplot(data=df, orient="h", palette="Set2")
    ax.plot()
    plt.show()
