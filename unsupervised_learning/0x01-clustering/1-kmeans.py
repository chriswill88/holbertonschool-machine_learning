#!/usr/bin/env python3
"""this modual contiains the function for task 1"""
import numpy as np
from sklearn.cluster import KMeans


def kmeans(X, k, iterations=1000):
    """
    initialize: implement Kmeans
    """
    try:
        if k < 1 or iterations < 1:
            return None, None
        n, d = X.shape

        mini = X.min(axis=0)
        maxi = X.max(axis=0)

        init = np.random.uniform(mini, maxi, (k, d))

        kmean = KMeans(k, init, 1, iterations).fit(X)
        C = kmean.cluster_centers_
        labels = kmean.labels_

        return C, labels
    except Exception:
        return None, None
