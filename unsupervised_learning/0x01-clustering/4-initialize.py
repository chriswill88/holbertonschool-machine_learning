#!/usr/bin/env python3
"""This is the modual contiains the function for task 4"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """This function initializes the variables for gaussian mixture model"""

    if not isinstance(X, np.ndarray) or len(X.shape()) != 2:
        return None, None, None
    if not isinstance(k, int) or k < 0:
        return None, None, None

    m, _ = kmeans(X, k)
    pi = np.ones((k)) / k
    d = X.shape[1]
    S = np.ones((k, d, d)) * np.identity(d)
    return pi, m, S
