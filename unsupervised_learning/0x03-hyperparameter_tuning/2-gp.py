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
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        kernel = np.sum(self.X**2, 1).reshape(-1, 1) + np.sum(
            self.X**2, 1) - 2 * np.dot(self.X, self.X.T)
        self.K = sigma_f**2 * np.exp(-0.5 / l**2 * kernel)

    def kernel(self, X1, X2):
        """
        kernel:
            @X1: is a numpy.ndarray of shape (m, 1)
            @X2: is a numpy.ndarray of shape (n, 1)
            Radial Basis Function (RBF)
        """
        kernel = np.sum(
            X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)
        return self.sigma_f**2 * np.exp(-0.5 / self.l**2 * kernel)

    def predict(self, X_s):
        """
        That predicts the mean and standard deviation of points in a Gaussian
        process:

        @X_s: w
        """
        B = self.kernel(X_s, self.X)
        C = self.kernel(self.X, self.X)
        A = self.kernel(X_s, X_s)
        D = self.kernel(self.X, X_s)

        mu = B @ np.linalg.inv(C) @ self.Y
        sigma = A - (B @ np.linalg.inv(C) @ D)
        return (mu.squeeze(), np.diag(sigma))

    def update(self, X_new, Y_new):
        """
            updates the variables for the class
        """
        self.X = np.append(self.X, X_new)
        self.Y = np.append(self.Y, Y_new)

        self.X = np.expand_dims(self.X, 1)
        self.Y = np.expand_dims(self.Y, 1)

        self.K = self.kernel(self.X, self.X)
