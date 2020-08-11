#!/usr/bin/env python3
"""This modual contains the function for task 3"""
import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
        That tests for the optimum number of clusters by variance:

        X is a numpy.ndarray of shape (n, d) containing the data set
        kmin is a positive integer containing the minimum number of clusters
         to check for (inclusive)
        kmax is a positive integer containing the maximum number of clusters
         to check for (inclusive)
        iterations is a positive integer containing the maximum number of
         iterations for K-means
        This function should analyze at least 2 different cluster sizes
    """

    results = []
    d_vars = []

    if kmax is None:
        kmax = X.shape[0]

    if not isinstance(kmin, int) or kmin < 1:
        return None, None
    if not isinstance(kmax, int) or kmax < 1 or kmin >= kmax:
        return None, None

    for k in range(kmin, kmax + 1):
        result = kmeans(X, k, iterations)
        results.append(result)
        d_vars.append(variance(X, result[0]))
    d_vars = [d_vars[0] - i for i in d_vars]
    return results, d_vars
