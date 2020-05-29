#!/usr/bin/env python3
"""that updates the weights and biases
of a neural network using gradient descent
with L2 regularization"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    updates the weights and biases of a neural network using gradient descent with L2 regularization:
    """
    m = Y.shape[1]
    A = cache["A{}".format(L)]
    DZ = A - Y
    for l in reversed(range(L)):
        w = weights["W{}".format(l + 1)]
        b = weights["b{}".format(l + 1)]
        A = cache["A{}".format(l + 1)]
        PreA = cache["A{}".format(l)]

        DW = (DZ @ PreA.T)/m + (lambtha/m*w)
        DB = np.sum(DZ, axis=1, keepdims=True)/m + (lambtha/m*w)
        da = w.T @ DZ
        DZ = da @ (A*(1-A))

        w -= alpha * DW
        b -= alpha * DB
