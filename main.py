import munich
import linear_regression
import statistics.corelation as corr
import statistics.outliers as outliers

def get_outliers_statistics(df):
    # outliers.boxplot(df, 'amenities')
    outliers.subplots(df)
    # outliers.scatter_plot(df, 'amenities', 'price')
    # outliers.pairplot(df)
    # outliers.cap_outliers(df['price'], 3)
    # outliers.displot(['guests_included'], df, 'price')

def train_with_linear_regression(df):
    train_columns = df.drop('price', axis=1)
    train_columns = train_columns.to_numpy()
    price_column = df.loc[:, 'price'].values
    return linear_regression.train(train_columns, price_column)

df_munich = munich.get_munich_data()
model = train_with_linear_regression(df_munich)
get_outliers_statistics(df_munich)
# corr.print_corelation(df_munich['amenities'].astype('int64'), df_munich['price'])
