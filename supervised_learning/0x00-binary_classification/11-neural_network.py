#!/usr/bin/env python3
"""this modual contains the class NeuralNetwork"""
import numpy as np


class NeuralNetwork:
    """
    neuralnetwork - this class defines a neural network with one hidden layer
    """
    def __init__(self, nx, nodes):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.randn(nodes, nx)
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.randn(1, nodes)
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        return self.__W1

    @property
    def b1(self):
        return self.__b1

    @property
    def A1(self):
        return self.__A1

    @property
    def W2(self):
        return self.__W2

    @property
    def b2(self):
        return self.__b2

    @property
    def A2(self):
        return self.__A2

    def forward_prop(self, X):
        """
        forward_prop figures out the
        forward propogation for the
        neural network with one hidden layer
        """
        W1 = self.__W1
        W2 = self.__W2
        b1 = self.__b1
        b2 = self.__b2

        Z1 = W1@X+b1
        A1 = 1/(1+np.exp(-1 * Z1))
        Z2 = W2@A1+b2
        A2 = 1/(1+np.exp(-1 * Z2))

        self.__A1, self.__A2 = A1, A2
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        loss = -1 * (Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = (1/m) * np.sum(loss)
        return cost
