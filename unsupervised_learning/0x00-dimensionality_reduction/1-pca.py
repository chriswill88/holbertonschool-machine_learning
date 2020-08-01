#!/usr/bin/env python3
"""this is the function for task 1"""
import numpy as np


def pca(X, ndim):
    """
    This function performs PCA on a dataset
    """
    X -= np.mean(X, axis=0)
    u, s, vh = np.linalg.svd(X)
    w = vh.T
    T = np.matmul(X, w[:, :ndim])
    return T
