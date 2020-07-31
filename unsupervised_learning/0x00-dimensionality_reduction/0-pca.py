#!/usr/bin/env python3
"""this is the function for task 0"""
import numpy as np


def pca(X, var=0.95):
    """
    This function performs PCA on a dataset
    """
    u, s, vh = np.linalg.svd(X)
    s = np.expand_dims(s, 1)
    c = np.cumsum(s)

    thresArr = c/sum(s)
    thresAxis = 0
    for i in thresArr:
        thresAxis += 1
        if i > var:
            break

    return vh.T[:, :thresAxis]  # Don't forget to transpose
