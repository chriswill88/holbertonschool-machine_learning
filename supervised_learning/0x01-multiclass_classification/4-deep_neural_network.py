#!/usr/bin/env python3
"""this modual contains the class DeepNeuralNetwork"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """defines a deep neural network performing binary classification"""
    def __init__(self, nx, layers, activation='sig'):
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
        if activation not in ['sig', 'tanh']:
            raise ValueError("activation must be 'sig' or 'tanh'")
        self.__activation = activation

    @property
    def activation(self):
        return self.__activation

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
        A = 0
        C["A0"] = X

        for layer in range(L):
            w = W["W{}".format(layer+1)]
            b = W["b{}".format(layer+1)]
            temp = X if layer == 0 else C["A{}".format(layer)]
            Z = w @ temp + b
            if layer == L - 1:
                t = np.exp(Z)
                NN = C["A{}".format(layer + 1)] = t/np.sum(t, axis=0)
            else:
                if self.__activation == 'sig':
                    NN = C["A{}".format(layer + 1)] = 1/(1+np.exp(-1 * Z))
                else:
                    NN = C["A{}".format(layer + 1)] = np.tanh(Z)

        return NN, C

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression"""
        m = Y.shape[1]
        loss = -1 * np.sum(Y * np.log(A))
        cost = (1/m) * loss
        return cost

    def evaluate(self, X, Y):
        """Evalueates neurons predictions"""
        NN, C = self.forward_prop(X)
        Ns = NN

        maxind = np.max(NN, axis=0)
        NN = np.where(NN == maxind, 1, 0)
        return NN, self.cost(Y, Ns)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """Calculates the gradient decent for a deep neural network"""
        W = self.__weights.copy()
        L = self.__L
        C = cache
        m = Y.shape[1]
        A = C["A{}".format(L)]
        act = self.__activation
        print(C)
        # depending on the activation we use slightly diffrent code
        da = -1 * (Y/A)+(1-Y)/(1-A)
        for lay in reversed(range(L)):
            w = W["W{}".format(lay + 1)]
            b = W["b{}".format(lay + 1)]
            A = C["A{}".format(lay + 1)]
            PreA = C["A{}".format(lay)]

            # derivative of activation
            if act == 'sig':
                DZ = da * (A*(1-A))
            else:
                DZ = da * (1 - A**2)

            DW = (DZ @ PreA.T)/m
            DB = np.sum(DZ, axis=1, keepdims=True)/m
            da = W["W{}".format(lay + 1)].T @ DZ

            self.__weights["W{}".format(lay + 1)] -= alpha * DW
            self.__weights["b{}".format(lay + 1)] -= alpha * DB

    def train(
            self, X, Y, iterations=5000,
            alpha=0.05, verbose=True,
            graph=True, step=100):
        """Trains a neural network"""
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
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        it = []
        cost = []
        for i in range(iterations):
            NN, C = self.forward_prop(X)
            if step == 0 or i % step == 0 or i == iterations:
                if verbose:
                    print("Cost after {} iterations: {}".format(i, c(Y, NN)))
                if graph:
                    it.append(i)
                    cost.append(self.cost(Y, NN))
            self.gradient_descent(Y, C, alpha)

        it = np.array(it)
        cost = np.array(cost)
        if graph:
            plt.plot(it, cost)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """pickles instances and saves them to a file"""
        if filename[-4:] != ".pkl":
            filename += ".pkl"
        with open(filename, 'wb') as file:
            pickle.dump(self, file)

    @staticmethod
    def load(filename):
        """loads pickeled instance"""
        try:
            with open(filename, 'rb') as file:
                return(pickle.load(file))
        except Exception:
            return None
