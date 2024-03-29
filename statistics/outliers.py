import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np

def get_outliers_statistics(df):
    # boxplot(df, 'amenities')
    subplots(df)
    # scatter_plot(df, 'amenities', 'price')
    # pairplot(df)
    # cap_outliers(df['price'], 3)
    # displot(['guests_included'], df, 'price')

def boxplot(dataframe, column):
    print('---------- outliers ----------')
    print(dataframe.sort_values(by=[column], ascending=False).head(100).to_string())
    sns.set_theme(style="whitegrid")
    sns.boxplot(x=dataframe[column])
    plt.show()
    print('---------- outliers end ----------')

def subplots(dataframe):
    ax = sns.boxplot(data=dataframe, orient="h", palette="Set2")
    ax.plot()
    plt.show()

def scatter_plot(dataframe, column1, column2):
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.scatter(dataframe[column1], dataframe[column2])
    ax.set_xlabel(column1)
    ax.set_ylabel(column2)
    plt.show()

def pairplot(dataframe):
    sns.pairplot(dataframe)
    plt.show()

def cap_outliers(column, zscore_threshold=2):
    median_val = column.median()
    mad_val = column.mad()
    st_dev = column.mad() * 1.4826
    z_score = (column - column.mean()) / st_dev
    outliers = abs(z_score) > zscore_threshold
    df_outliers = column.loc[outliers]
    df_outliers = column[column.index.isin(df_outliers.index)]
    print(df_outliers.to_string())

def displot(column_array, dataframe, target='price'):
    for column in column_array:
        sns.displot(dataframe, x=column, hue=target, height=10, aspect=2, multiple="dodge")
        plt.show()