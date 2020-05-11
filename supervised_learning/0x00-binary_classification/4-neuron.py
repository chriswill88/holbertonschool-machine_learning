#!/usr/bin/env python3
"""This modual holds the class created for task 3"""
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
        """
        This function calculates the forward propogation
        and updates the private attibute A
        """
        # X numpy array size = (nx - , m - )
        Z = np.dot(self.W, X) + self.b
        # Z = the (weight*activation)+bias for all data in the set

        A = 1/(1 + np.exp(-1 * Z))
        # applying the sigmoid function to Z (3brown1blue need to rewatch)

        self.__A = A
        return self.__A

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        loss = -1 * (Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        cost = (1/m) * np.sum(loss)
        return cost

    def evaluate(self, X, Y):
        """Evalueates neurons predictions"""
        predict = self.forward_prop(X)
        predict = np.where(predict < 0.5, 0, 1)
        return predict, self.cost(Y, self.__A)
