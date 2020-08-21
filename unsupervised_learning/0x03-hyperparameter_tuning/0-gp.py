#!/usr/bin/env python3
"""This class is for task 0"""
import numpy as np


class GaussianProcess:
    """
    Gaussian Process:
        @X_init is a numpy.ndarray of shape (t, 1) representing the inputs
         already sampled with the black-box function
        @Y_init is a numpy.ndarray of shape (t, 1) representing the outputs
         of the black-box function for each input in X_init
        @t is the number of initial samples
        @l is the length parameter for the kernel
        @sigma_f is the standard deviation given to the output of the
         black-box function
    """
    def __init__(self, X_init, Y_init, L=1, sigma_f=1):
        self.X = X_init
        self.Y = Y_init
        self.L = L
        self.sigma_f = sigma_f
        sqdist = np.sum(self.X**2, 1).reshape(-1, 1) + np.sum(
            self.X**2, 1) - 2 * np.dot(self.X, self.X.T)
        self.K = sigma_f**2 * np.exp(-0.5 / L**2 * sqdist)

    def kernel(self, X1, X2):
        """
        kernel:
            @X1: is a numpy.ndarray of shape (m, 1)
            @X2: is a numpy.ndarray of shape (n, 1)
            Radial Basis Function (RBF)
        """
        kernel = np.sum(
            X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.L**2 * kernel)
