import numpy as np
from algorithms import regression_impl as reg
from algorithms.neural_network.network import Network
from algorithms.neural_network.fc_layer import FCLayer
from algorithms.neural_network.activation_layer import ActivationLayer
from algorithms.neural_network.activations import tanh, tanh_prime, relu, relu_prime
from algorithms.neural_network.losses import mse, mse_prime

def extract_df(df):
    price_column = df.loc[:, 'price'].values
    train_columns = df.drop('price', axis=1)
    train_columns = train_columns.to_numpy()
    return train_columns, price_column

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_der(x):
    return sigmoid(x)*(1-sigmoid(x))


def train(dataframe):
    x, y = extract_df(dataframe)
    x_train, y_train, x_test, y_test, x_val, y_val = reg.split_dataset(x, y)
    y_train = y_train.reshape(-1, 1)
    x_shape, y_shape = x_train.shape

    net = Network()
    net.add(FCLayer(21, 100))
    net.add(ActivationLayer(relu, relu_prime))
    net.add(FCLayer(100, 1))
    net.add(ActivationLayer(relu, relu_prime))

    # train
    net.use(mse, mse_prime)
    net.fit(x_train, y_train, epochs=1000, learning_rate=0.1)

    # test
    out = net.predict(x_train)
    print(out)