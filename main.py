import munich
import berlin
import linear_regression
import statistics.outliers as outliers
import statistics.corelation as corelation
import regression_optimization

def train_with_linear_regression(df):
    train_columns = df.drop('price', axis=1)
    train_columns = train_columns.to_numpy()
    price_column = df.loc[:, 'price'].values
    return linear_regression.train(train_columns, price_column)


df_munich = munich.get_munich_data()
model = train_with_linear_regression(df_munich)
# outliers.get_outliers_statistics(df_munich)
# corelation.print_corelation(df_munich['day_of_week'].head(20).astype('int64'), df_munich['price'].head(20))
#df_berlin = berlin.get_berlin_dataset()
#model = train_with_linear_regression(df_berlin)