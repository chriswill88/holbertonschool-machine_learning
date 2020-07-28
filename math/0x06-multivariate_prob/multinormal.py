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
        d, n = data.shape
        if n < 2:
            raise ValueError("data must contain multiple data points")
        mean = np.mean(data, 1)
        self.mean = np.expand_dims(mean, 1)  # Always Check Dimensions
        data -= self.mean
        self.cov = np.matmul(data, data.T)/(n - 1)
