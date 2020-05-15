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
            self.weights["W{}".format(l+1)] = n(lay[l], il)*np.sqrt(2/il)
            self.weights["b{}".format(l+1)] = np.zeros((lay[l], 1))

    @property
    def L(self):
        return self.__L

    @property
    def cache(self):
        return self.__cache

    @property
    def weights(self):
        return self.__weights
