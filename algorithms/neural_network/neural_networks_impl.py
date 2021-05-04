import numpy as np
from algorithms import regression_impl as reg
from algorithms.neural_network.network import Network
from algorithms.neural_network.fc_layer import FCLayer
from algorithms.neural_network.activation_layer import ActivationLayer
from algorithms.neural_network.activations import linear, linear_prime, relu, relu_prime
from algorithms.neural_network.losses import mse, mse_prime
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

def extract_df(df):
    price_column = df.loc[:, 'price'].values
    train_columns = df.drop('price', axis=1)
    train_columns = train_columns.to_numpy()
    return train_columns, price_column

def train(dataframe, print_stats=True):
    x, y = extract_df(dataframe)
    x_train, y_train, x_test, y_test, x_val, y_val = reg.split_dataset(x, y)
    y_train = y_train.reshape(-1, 1)
    x_shape, y_shape = x_train.shape

    net = Network()
    net.add(FCLayer(y_shape, 20))
    net.add(ActivationLayer(relu, relu_prime))
    net.add(FCLayer(20, 5))
    net.add(ActivationLayer(relu, relu_prime))
    net.add(FCLayer(5, 1))
    net.add(ActivationLayer(linear, linear_prime))

    net.use(mse, mse_prime)
    net.fit(x_train, y_train, epochs=1000, learning_rate=0.0001)

    # test
    out = net.predict(x_test)
    out = [x[0][0] for x in out]

    if print_stats:
        r_sq = r2_score(y_test, out)
        rmse = np.sqrt(mean_squared_error(y_test, out))
        mae = mean_absolute_error(y_test, out)
        print("coefficient of determination:: %f" % r_sq)
        print("RMSE: %f" % rmse)
        print("MAE: %f" % mae)