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

iris = datasets.load_iris()
X = iris.data
Y = iris.target

train_ratio = 0.70
test_ratio = 0.30

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
def train(dataframe, to_predict=True, print_stats=True):
    x, y = extract_df(dataframe)
    x_train, y_train, x_test, y_test, x_val, y_val = reg.split_dataset(x,y)
    x_train2, y_train2, x_test2, y_test2 = split_dataset(x,y)
    svc = SVC(probability=True, kernel='linear')

    rng = np.random.RandomState(1)
    print("XTRAIN")
    print(x_train2.shape)
    print(x_train2)
    print("YTRAIN")
    print(y_train2.shape)
    print(y_train2)
    print("x_test2")
    print(x_test2.shape)
    print(x_test2)
    print("y_test2")
    print(y_test2.shape)
    print(y_test2)

    regr_1 = DecisionTreeRegressor(max_depth=4)
    regr_2 = AdaBoostRegressor(DecisionTreeRegressor(max_depth=4),n_estimators=300, random_state=rng)

    abc = AdaBoostClassifier(n_estimators=100, random_state=0, learning_rate=1)
    # model = abc.fit(x_train2, y_train2)
    regr_1.fit(x_train2, y_train2)
    regr_2.fit(x_train2, y_train2)

    y_1 = regr_1.predict(x_test2)
    y_2 = regr_2.predict(x_test2)

    model1 = regr_1.fit(x_train2, y_train2)
    model2 = regr_2.fit(x_train2, y_train2)
    # y_pred = model.predict(x_test);
    clf = RandomForestRegressor(n_estimators=10)
    # y_pred2 = model.predict(x_test)

    print("Y_PRED")
    print(y_2)
    print(y_2.shape)

    # print("Accuracy:", model2.score(y_test2.reshape(-1,1), y_2.reshape(-1,1)))





