#!/usr/bin/env python3
"""that updates the weights and biases
of a neural network using gradient descent
with L2 regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
        updates the weights and biases of a neural network using gradient descent with L2 regularization:

        @Y is a one-hot numpy.ndarray of shape (classes, m) that contains the correct labels for the data
        @@classes is the number of classes
        @m is the number of data points
        @weights is a dictionary of the weights and biases of the neural network
        @cache is a dictionary of the outputs of each layer of the neural network
        @alpha is the learning rate
        @lambtha is the L2 regularization parameter
        @L is the number of layers of the network

        The neural network uses tanh activations on each layer except the last, which uses a softmax activation
        The weights and biases of the network should be updated in place
    """

    m = Y.shape[1]
    A = cache["A{}".format(L)]
    for lay in reversed(range(L)):
        w = weights["W{}".format(lay + 1)]
        b = weights["b{}".format(lay + 1)]
        PreA = cache["A{}".format(lay)]

        if lay == L - 1:
                DZ = A - Y
        else:
            A = cache["A{}".format(lay + 1)]
            DZ = da * (1 - A**2)

        DW = (DZ @ PreA.T)/m
        DB = np.sum(DZ, axis=1, keepdims=True)/m
        da = weights["W{}".format(lay + 1)].T @ DZ

        weights["W{}".format(lay + 1)] -= alpha * DW
        weights["b{}".format(lay + 1)] -= alpha * DB