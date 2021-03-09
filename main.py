import munich
import linear_regression

def train_with_linear_regression(df):
    train_columns = df.drop('price', axis=1)
    train_columns = train_columns.to_numpy()
    price_column = df.loc[:, 'price'].values
    return linear_regression.train(train_columns, price_column)

df_munich = munich.get_munich_data()
model = train_with_linear_regression(df_munich)