import munich
import linear_regression
import pandas as pd
import statistics.corelation as corr
import statistics.outliers as outliers

def train_with_linear_regression(df):
    train_columns = df.drop('price', axis=1)
    train_columns = train_columns.to_numpy()
    price_column = df.loc[:, 'price'].values
    return linear_regression.train(train_columns, price_column)

df_munich = munich.get_munich_data()
model = train_with_linear_regression(df_munich)
# corr.print_corelation(df_munich['amenities'].astype('int64'), df_munich['price'])
# outliers.boxplot(df_munich, 'amenities')
outliers.subplots(df_munich)
# outliers.scatter_plot(df_munich, 'number_of_reviews', 'price')
# outliers.pairplot(df_munich)
outliers.cap_outliers(df_munich['price'], 3)
# outliers.displot(['guests_included'], df_munich, 'price')