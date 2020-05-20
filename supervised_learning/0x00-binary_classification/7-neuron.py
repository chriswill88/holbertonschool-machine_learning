#!/usr/bin/env python3
"""This modual holds the class created for task 3"""
import numpy as np
import matplotlib.pyplot as plt


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

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """
            - Calculates one pass of gradient descent on the neuron
            - Gradient decent updates the weights and biases
        """
        m = Y.shape[1]
        W = self.__W
        b = self.__b

        Dz = A - Y
        Dw = (1/m) * (Dz @ X.T)
        Db = (1/m) * np.sum(Dz)

        self.__W = W - alpha * Dw
        self.__b = b - alpha * Db

    def train(
            self, X, Y,
            iterations=5000, alpha=0.05, verbose=True,
            graph=True, step=100):
        """Trains a neuron"""
        c = self.cost
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        if iterations < 0:
            raise ValueError("iterations must be a positive integer")
        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha < 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if not isinstance(step, int):
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        it = []
        cost = []
        for i in range(iterations + 1):
            A = self.forward_prop(X)
            if step == 0 or i % step == 0:
                if verbose:
                    print("Cost after {} iterations: {}".format(i, c(Y, A)))
                if graph:
                    it.append(i)
                    cost.append(self.cost(Y, A))
            self.gradient_descent(X, Y, A, alpha)

        it = np.array(it)
        cost = np.array(cost)
        if graph:
            plt.plot(it, cost)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
