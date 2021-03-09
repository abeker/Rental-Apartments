import numpy as np
from sklearn.linear_model import LinearRegression

def print_statistics(model, x, y):
    r_sq = model.score(x, y)
    print('coefficient of determination:', r_sq)
    print('intercept:', model.intercept_)
    print('slope:', model.coef_)

def train(x, y):
    x, y = np.array(x), np.array(y)
    model = LinearRegression().fit(x, y)
    return model

def predict(model, x):
    y_pred = model.predict(np.array(x))
    return y_pred