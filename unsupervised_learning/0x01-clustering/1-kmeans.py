#!/usr/bin/env python3
"""this modual contiains the function for task 1"""
import numpy as np


def kmeans(X, k, iterations=1000):
    """
    initialize: implement Kmeans
    """
    if not isinstance(iterations, int) or iterations < 1:
        return None, None
    try:
        n, d = X.shape

        mini = X.min(axis=0)
        maxi = X.max(axis=0)

        init = np.random.uniform(mini, maxi, (k, d))
        kLabels = np.zeros(X.shape[0],)
        while iterations:
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
            if compare.all():
                break
            init = centroid
            iterations -= 1

        dist = np.sqrt(((X - centroid[:, np.newaxis])**2).sum(axis=2))
        kLabels = np.argmin(dist, 0)
        return centroid, kLabels
    except Exception:
        return None, None
