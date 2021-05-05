import numpy as np

# activation function and its derivative
def tanh(x):
    return np.tanh(x)

def tanh_prime(x):
    return 1-np.tanh(x)**2

def relu(x):
    return x * (x > 0)

def relu_prime(x):
    return 1. * (x > 0)

def linear(x):
    return x

def linear_prime(x):
    return 1

def sigmoid(x):
    return 1/(1+np.exp(-x))

def sigmoid_prime(x):
    return sigmoid(x)*(1-sigmoid(x))