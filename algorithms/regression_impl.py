import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split
import utility.enums as enum

# ridge and lasso regression are optimization algorithm of linear regression.
# elastic net is combination of ridge and lasso

train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15

def print_statistics(model, x, y):
    r_sq = model.score(x, y)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)

def split_dataset(x, y):
    x_in, y_in = np.array(x), np.array(y)
    x_remaining, x_test, y_remaining, y_test = train_test_split(x_in, y_in, test_size=test_ratio)
    ratio_remaining = 1 - test_ratio
    ratio_val_adjusted = test_ratio / ratio_remaining
    x_train, x_val, y_train, y_val = train_test_split(x_remaining, y_remaining, test_size=ratio_val_adjusted)
    return x_train, y_train, x_test, y_test, x_val, y_val
    # return x_train, y_train

def train(df, type_of_regression):
    train_columns = df.drop('price', axis=1)
    train_columns = train_columns.to_numpy()
    price_column = df.loc[:, 'price'].values
    if type_of_regression == enum.RegressionType.LINEAR:
        return train_linear(train_columns, price_column)
    elif type_of_regression == enum.RegressionType.RIDGE:
        return train_ridge(train_columns, price_column)
    elif type_of_regression == enum.RegressionType.LASSO:
        return train_lasso(train_columns, price_column)

def train_ridge(x, y):
    x_train, y_train, x_test, y_test, x_val, y_val = split_dataset(x, y)
    ridge_model = Ridge(normalize=True)
    ridge_model = ridge_model.fit(x_train, y_train)
    print_statistics(ridge_model, x_train, y_train)
    return ridge_model

def train_lasso(x, y):
    x_train, y_train, x_test, y_test, x_val, y_val = split_dataset(x, y)
    lasso_model = Lasso(normalize=True).fit(x_train, y_train)
    print_statistics(lasso_model, x_train, y_train)
    return lasso_model

def train_linear(x, y):
    x, y = np.array(x), np.array(y)
    model = LinearRegression().fit(x, y)
    print_statistics(model, x, y)
    return model

def predict_linear(model, x):
    y_pred = model.predict(np.array(x))
    return y_pred