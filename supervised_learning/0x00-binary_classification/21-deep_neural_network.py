#!/usr/bin/env python3
"""this modual contains the class DeepNeuralNetwork"""
import numpy as np


class DeepNeuralNetwork:
    """defines a deep neural network performing binary classification"""
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list):
            raise TypeError("layers must be a list of positive integers")
        if len(layers) == 0 or min(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        self.__L = len(layers)  # the number of layers
        self.__cache = {}  # to hold all intermediary values of the network.
        self.__weights = {}  # holds all weights and bias
        n = np.random.randn
        for l in range(self.L):
            lay = layers
            il = nx if (l < 1) else lay[l-1]
            self.__weights["W{}".format(l+1)] = n(lay[l], il)*np.sqrt(2/il)
            self.__weights["b{}".format(l+1)] = np.zeros((lay[l], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network"""
        W = self.__weights
        L = self.__L
        C = self.__cache
        C["A0"] = X

        for layer in range(L):
            w = W["W{}".format(layer+1)]
            b = W["b{}".format(layer+1)]

            temp = X if layer == 0 else C["A{}".format(layer)]
            Z = w @ temp + b
            NN = C["A{}".format(layer + 1)] = 1/(1+np.exp(-1 * Z))

        return NN, C

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        loss = -1 * (Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = (1/m) * np.sum(loss)
        return cost

    def evaluate(self, X, Y):
        """Evalueates neurons predictions"""
        NN, C = self.forward_prop(X)
        Ns = NN
        NN = np.where(NN < 0.5, 0, 1)
        return NN, self.cost(Y, Ns)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates the gradient decent for a deep neural network"""
        W = self.__weights
        L = self.__L
        C = cache
        m = Y.shape[1]
        A = C["A{}".format(L)]
        da = -1 * (Y/A)+(1-Y)/(1-A)
        for l in reversed(range(L)):
            w = W["W{}".format(l + 1)]
            b = W["b{}".format(l + 1)]
            A = C["A{}".format(l + 1)]
            PreA = C["A{}".format(l)]

            DZ = da * (A*(1-A))
            DW = (DZ @ PreA.T)/m
            DB = np.sum(DZ, axis=1, keepdims=True)/m
            da = w.T @ DZ

            w -= alpha * DW
            b -= alpha * DB
