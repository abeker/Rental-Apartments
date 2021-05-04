# Load libraries
import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.svm import SVC
from algorithms import regression_impl as reg
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing

iris = datasets.load_iris()
X = iris.data
Y = iris.target
# print(X.shape)
# print(Y.shape)

train_ratio = 0.70
test_ratio = 0.30

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

def extract_df(df):
    price_column = df.loc[:, 'price'].values
    train_columns = df.drop('price', axis=1)
    train_columns = train_columns.to_numpy()
    return train_columns, price_column

def split_dataset(x, y):
    x_in, y_in = np.array(x), np.array(y)
    x_train, x_test, y_train, y_test = train_test_split(x_in, y_in, test_size=test_ratio)
    return x_train, y_train, x_test, y_test


# def train_sample(params, dtrain, num_boost_round):
def train(dataframe):
    train_columns = dataframe.drop('price', axis=1)
    train_columns = train_columns.to_numpy()
    price_column = dataframe.loc[:, 'price'].values
    x, y = extract_df(dataframe)
    # print(x.shape)
    # print(y.shape)
    x_train, y_train, x_test, y_test, x_val, y_val = reg.split_dataset(x,y)
    # lab_enc = preprocessing.LabelEncoder()
    # x = lab_enc.fit_transform(x)

    print("PRICE")
    print(price_column)
    print(price_column.shape)
    # x_train, y_train, x_test, y_test = split_dataset(train_columns, price_column)
    print("TRAIN")
    print(x_train)
    print(x_train.shape)
    print(y_train.shape)
    svc = SVC(probability=True, kernel='linear')

    # regr_1 = DecisionTreeRegressor(max_depth=4)
    regr_2 = AdaBoostRegressor(n_estimators=300, random_state=0)

    abc = AdaBoostClassifier()
    # model = abc.fit(x_train, y_train)
    # regr_1.fit(x_train, y_train)
    regr_2.fit(x_train, y_train)

    # y_1 = regr_1.predict(x_test)
    # y_2 = regr_2.predict(x_test)

    # model1 = regr_1.fit(x_train, y_train)
    # model2 = regr_2.fit(x_train, y_train)
    # y_pred = model.predict(x_test);
    clf = RandomForestRegressor(n_estimators=10)
    y_pred2 = regr_2.predict(x_test)

    print("Accuracy Square:", abs(metrics.r2_score(y_test, y_pred2)))
    print("mean square log error:", metrics.mean_squared_log_error(y_test,y_pred2))





