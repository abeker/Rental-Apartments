import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics


def prepare_data(dataset):
   # divides data into attributes and labels:
   count_col = dataset.shape[1]
   X = dataset.iloc[:, 1:count_col].values
   y = dataset.iloc[:, 0].values
   #divide the data into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
   # Feature Scaling
   #print(X_test)
   sc = StandardScaler()
   X_train = sc.fit_transform(X_train)
   X_test = sc.transform(X_test)
   #print(X_test)
   return X_train, X_test, y_train, y_test


def train(dataset):
    print("train model")
    X_train, X_test, y_train, y_test = prepare_data(dataset)
    regressor = RandomForestRegressor(n_estimators=20, random_state=0)
    regressor.fit(X_train, y_train)
    y_pred = regressor.predict(X_test)
    print('Mean Absolute Error-MAE:', metrics.mean_absolute_error(y_test, y_pred))
    #print('Mean Squared Error:', metrics.mean_squared_error(y_test, y_pred))
    print('Root Mean Squared Error-RMSE:', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
