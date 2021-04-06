import pandas as pd
import scipy.stats
import matplotlib.pyplot as plt
import seaborn as sns

def plot(x, y, x_label_name, y_label_name):
    slope, intercept, r, p, stderr = scipy.stats.linregress(x, y)
    line = f'Regression line: y={intercept:.2f}+{slope:.2f}x, r={r:.2f}'
    fig, ax = plt.subplots()
    ax.plot(x, y, linewidth=0, marker='s', label='Data points')
    ax.plot(x, intercept + slope * x, label=line)
    ax.set_xlabel(x_label_name)
    ax.set_ylabel(y_label_name)
    ax.legend(facecolor='white')
    plt.show()

def plot_histogram(data, title, y_label_name):
    data.plot.hist(grid=True, bins=20, rwidth=0.9, color='#607c8e')
    plt.title(title)
    plt.xlabel('Counts')
    plt.ylabel(y_label_name)
    plt.grid(axis='y', alpha=0.75)
    plt.show()

def print_corelation(x_values, y_values):
    print('----------corelation----------')
    x = pd.Series(x_values)
    y = pd.Series(y_values)
    plot(x, y, 'amenities', 'price')
    plot_histogram(y, 'Price Histogram', 'price')
    print('Pearson’s: ', x.corr(y))
    print('Spearman’s: ', x.corr(y, method='spearman'))
    print('Kendall’s: ', x.corr(y, method='kendall'))
    print('----------corelation end----------')

def print_df_corelation(df):
    corrmat = df.corr()
    top_corr_features = corrmat.index
    fig, ax = plt.subplots(figsize=(30, 30))
    # plot heat map
    heatmap = sns.heatmap(df[top_corr_features].corr(), vmin=-1, vmax=1, annot=True, cmap="BrBG", linewidths=1, ax=ax)
    heatmap.set_title('Correlation Heatmap', fontdict={'fontsize': 15}, pad=15)
    plt.show()