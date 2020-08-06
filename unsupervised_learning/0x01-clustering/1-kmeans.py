#!/usr/bin/env python3
"""this modual contiains the function for task 1"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    initialize: implement Kmeans
    """
    if k < 1 or iterations < 1:
        return None, None
    n, d = X.shape

    mini = X.min(axis=0)
    maxi = X.max(axis=0)

    init = np.random.uniform(mini, maxi, (k, d))
    return recurKmean(iterations, init, X)


def recurKmean(iterations, init, X):
    """uses recursion to calculate kmean"""
    kLabels = np.zeros(X.shape[0],)
    centroid = init.copy()
    dist = np.sqrt(((X - centroid[:, np.newaxis])**2).sum(axis=2))
    kLabels = np.argmin(dist, 0)
    for c3, ele3 in enumerate(init):
        cor = X[np.where(kLabels == c3)]
        if len(cor) > 0:
            new = np.mean(cor, 0)
        else:
            new = np.random.uniform(
                X.min(axis=0), X.max(axis=0), (1, X.shape[1]))
        centroid[c3] = new
    compare = centroid == init
    if not compare.all() and iterations > 1:
        return recurKmean(iterations - 1, centroid, X)
    return centroid, kLabels
