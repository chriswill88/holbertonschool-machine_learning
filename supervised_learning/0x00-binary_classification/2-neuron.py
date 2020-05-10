#!/usr/bin/env python3
"""This modual holds the class created for task 1"""
import numpy as np


class Neuron:
    """
        Neuron - class for a neuron
        nx = the number of input freatures to the neuron
    """
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A

    def forward_prop(self, X):
        # X numpy array size = (nx - , m - )
        Z = np.dot(self.W, X) + self.b
        # Z = the (weight*activation)+bias for all data in the set

        A = 1/(1 + np.exp(-1 * Z))
        # applying the sigmoid function to Z (3brown1blue need to rewatch)

        self.__A = A
        return self.__A
