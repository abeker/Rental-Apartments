import munich
import berlin
import linear_regression
import regression_optimization
import statistics.corelation as corr

def train_with_linear_regression(df):
    train_columns = df.drop('price', axis=1)
    train_columns = train_columns.to_numpy()
    price_column = df.loc[:, 'price'].values
    return linear_regression.train(train_columns, price_column)


df_munich = munich.get_munich_data()
model = train_with_linear_regression(df_munich)

#df_berlin = berlin.get_berlin_dataset()
#model = train_with_linear_regression(df_berlin)

corr.print_corelation(df_munich['amenities'].astype('int64'), df_munich['price'])