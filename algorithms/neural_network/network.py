import numpy as np

class Network:
    def __init__(self):
        self.layers = []
        self.loss = None
        self.loss_prime = None

    # add layer to network
    def add(self, layer):
        self.layers.append(layer)

    # set loss to use
    def use(self, loss, loss_prime):
        self.loss = loss
        self.loss_prime = loss_prime

    # predict output for given input
    def predict(self, input_data):
        # sample dimension first
        samples = len(input_data)
        result = []

        # run network over all samples
        for i in range(samples):
            # forward propagation
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    # train the network
    def fit(self, x_train, y_train, epochs, learning_rate, early_stop_err):
        # sample dimension first
        samples = len(x_train)
        previous_err = 0

        # training loop
        for epoch in range(epochs):
            err = 0
            for sample_num in range(samples):
                # forward propagation
                output = x_train[sample_num]
                output = np.reshape(output, (1, output.size))
                for layer in self.layers:
                    output = layer.forward_propagation(output)

                # compute loss (for display purpose only)
                err += self.loss(y_train[sample_num], output)

                # backward propagation
                error = self.loss_prime(y_train[sample_num], output)
                for layer in reversed(self.layers):
                    error = layer.backward_propagation(error, learning_rate)

            # calculate average error on all samples
            err = err / samples
            if abs(previous_err - err) < early_stop_err:
                break
            previous_err = err
            print('epoch %d/%d   error=%f' % (epoch+1, epochs, err))