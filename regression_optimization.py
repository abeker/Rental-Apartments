import numpy as np
import linear_regression
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.model_selection import train_test_split

#ridge and lasso regression are optimization algorithm of linear regression.
#elastic net is combination of ridge and lasso

train_ratio = 0.70
validation_ratio = 0.15
test_ratio = 0.15

def split_dataset(x, y):
    X, Y = np.array(x), np.array(y)
    x_remaining, x_test, y_remaining, y_test = train_test_split(X, Y, test_size=test_ratio)
    ratio_remaining = 1 - test_ratio
    ratio_val_adjusted = test_ratio / ratio_remaining
    x_train, x_val, y_train, y_val = train_test_split(x_remaining, y_remaining, test_size=ratio_val_adjusted)
    #return x_train, y_train, x_test, y_test, x_val, y_val
    return x_train, y_train


def train_ridge_regression(x, y):
    x_train, y_train = split_dataset(x, y)
    ridge_model = Ridge(normalize= True)
    ridge_model = ridge_model.fit(x_train, y_train)
    linear_regression.print_statistics(ridge_model, x_train, y_train)
    return ridge_model


def train_lasso_regression(x, y):
    x_train, y_train = split_dataset(x, y)
    lasso_model = Lasso(normalize= True).fit(x_train, y_train)
    linear_regression.print_statistics(lasso_model, x_train, y_train)
    return lasso_model






