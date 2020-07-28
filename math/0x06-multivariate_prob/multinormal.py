#!/usr/bin/env python3
"""this module contains the class multinormal"""
import numpy as np


class MultiNormal:
    """
    Multivariate Normal distribution
    """
    def __init__(self, data):
        if not isinstance(data, np.ndarray) or len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        self.d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        mean = np.mean(data, 1)
        self.mean = np.expand_dims(mean, 1)  # Always Check Dimensions
        data -= self.mean
        self.cov = np.matmul(data, data.T)/(n - 1)

    def pdf(self, x):
        """
        Calculates the PDF at a data point:
        """
        if not isinstance(x, np.ndarray):
            raise TypeError("x must be a numpy.ndarray")
        if len(x.shape) != 2 or x.shape != (self.d, 1):
            raise ValueError("x must have the shape ({d}, 1)".format(self.d))
        det = np.linalg.det(self.cov)
        b = (((2 * np.pi)**self.d) * det)**(.5)
        m = (x - self.mean)
        i = np.linalg.inv(self.cov)
        return ((1/b) * np.exp((-1/2) * m.T @ i @ m))[0][0]
