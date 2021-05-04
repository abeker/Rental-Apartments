from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from algorithms import regression_impl as reg
import numpy as np
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV
from datetime import datetime

def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))

def add_params():
    param = {'max_depth': 2,
             'eta': 1,
             'objective': 'binary:logistic',
             'eval_metric': 'auc',
             'nthread': 4}
    return param

def extract_df(df):
    price_column = df.loc[:, 'price'].values
    train_columns = df.drop('price', axis=1)
    train_columns = train_columns.to_numpy()
    return train_columns, price_column

def train_sample(params, dtrain, num_boost_round):
    gridsearch_params = [
        (subsample, colsample)
        for subsample in [i / 10. for i in range(2, 11)]
        for colsample in [i / 10. for i in range(2, 11)]
    ]
    min_mae = float("Inf")
    best_params = None
    # We start by the largest values and go down to the smallest
    for subsample, colsample in reversed(gridsearch_params):
        print("CV with subsample={}, colsample={}".format(
            subsample,
            colsample))
        # We update our parameters
        params['subsample'] = subsample
        params['colsample_bytree'] = colsample
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics='mae',
            early_stopping_rounds=10
        )
        # Update best score
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = (subsample, colsample)
    print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

def train_params(params, dtrain, num_boost_round):
    gridsearch_params = [
        (max_depth, min_child_weight)
        for max_depth in range(3, 12)
        for min_child_weight in range(5, 8)
    ]
    min_mae = float("Inf")
    best_params = None
    for max_depth, min_child_weight in gridsearch_params:
        print("CV with max_depth={}, min_child_weight={}".format(max_depth, min_child_weight))
        # Update our parameters
        params['max_depth'] = max_depth
        params['min_child_weight'] = min_child_weight
        # Run CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics='mae',
            early_stopping_rounds=10
        )
        # Update best MAE
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        print("\tMAE {} for {} rounds".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = (max_depth, min_child_weight)
    print("Best params: {}, {}, MAE: {}".format(best_params[0], best_params[1], min_mae))

def train_eta(params, dtrain, num_boost_round):
    min_mae = float("Inf")
    best_params = None
    for eta in [.3, .2, .1, .05, .01, .005]:
        print("CV with eta={}".format(eta))
        # We update our parameters
        params['eta'] = eta
        # Run and time CV
        cv_results = xgb.cv(
            params,
            dtrain,
            num_boost_round=num_boost_round,
            seed=42,
            nfold=5,
            metrics=['mae'],
            early_stopping_rounds=10
        )
        # Update best score
        mean_mae = cv_results['test-mae-mean'].min()
        boost_rounds = cv_results['test-mae-mean'].argmin()
        print("\tMAE {} for {} rounds\n".format(mean_mae, boost_rounds))
        if mean_mae < min_mae:
            min_mae = mean_mae
            best_params = eta
    print("Best params: {}, MAE: {}".format(best_params, min_mae))

def get_optimal_params(x, y):
    classifier = xgb.XGBClassifier()
    params = {
        "learning_rate": [0.25, 0.30],
        "max_depth": [3, 4],
        "min_child_weight": [1, 3],
        "gamma": [0.0, 0.1],
        "colsample_bytree": [0.3, 0.4]
    }
    random_search = RandomizedSearchCV(classifier, param_distributions=params, n_iter=5, scoring='roc_auc', n_jobs=-1,
                                       cv=5, verbose=3)
    start_time = timer(None)  # timing starts from this point for "start_time" variable
    print(start_time)
    random_search.fit(x, y)
    print(timer(start_time))  # timing ends here for "start_time" variable
    print(random_search.best_estimator_)
    print(random_search.best_params_)

def train(dataframe, to_predict=True, print_stats=True):
    x, y = extract_df(dataframe)
    x_train, y_train, x_test, y_test, x_val, y_val = reg.split_dataset(x, y)

    train_mat = xgb.DMatrix(x_train, label=y_train)
    test_mat = xgb.DMatrix(x_test, label=y_test)

    params = {
        'max_depth': 7,
        'min_child_weight': 6,
        'eta': 0.05,
        'subsample': 1,
        'colsample_bytree': 1,
        'eval_metric': 'mae',
        'gamma': 0
    }
    num_boost_round = 999
    model = xgb.train(
        params,
        train_mat,
        num_boost_round=num_boost_round,
        evals=[(test_mat, "Test")],
        early_stopping_rounds=20
    )
    print("Best MAE: {:.2f} in {} rounds".format(model.best_score, model.best_iteration + 1))

    if to_predict:
        predictions = model.predict(test_mat)
        if print_stats:
            rmse = np.sqrt(mean_squared_error(y_test, predictions))
            mae = mean_absolute_error(model.predict(test_mat), y_test)
            r_sq = r2_score(y_test, predictions)
            print("coefficient of determination:: %f" % r_sq)
            print("RMSE: %f" % rmse)
            print("MAE: %f" % mae)
        return predictions