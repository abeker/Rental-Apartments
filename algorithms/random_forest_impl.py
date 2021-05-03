import numpy as np
from algorithms import regression_impl as reg
from statistics import results as res
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


def prepare_data(dataset):
   # divides data into attributes and labels:
   count_col = dataset.shape[1]
   X = dataset.iloc[:, 1:count_col].values
   y = dataset.iloc[:, 0].values
   #divide the data into training and testing sets
   X_train, y_train, X_test, y_test, x_val, y_val = reg.split_dataset(X, y)
   # Feature Scaling
   sc = StandardScaler()
   X_train = sc.fit_transform(X_train)
   X_test = sc.transform(X_test)
   return X_train, X_test, y_train, y_test

def print_statistic(model, X_train, y_train, y_test, y_pred):
    r_sq = model.score(X_train, y_train)
    print('coefficient of determination:', r_sq)
    print('Mean Absolute Error-MAE:', metrics.mean_absolute_error(y_test, y_pred))
    print('Root Mean Squared Error-RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    res.prediction_plot(y_test, y_pred)


def train(dataset):
    print("**********RANDOM FOREST*************")
    X_train, X_test, y_train, y_test = prepare_data(dataset)
    regressor = RandomForestRegressor(n_estimators=100, random_state=0, min_impurity_decrease=0, min_samples_leaf=1)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    print_statistic(regressor, X_train, y_train, y_test, y_pred)
