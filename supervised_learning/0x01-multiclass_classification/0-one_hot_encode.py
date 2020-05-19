#!/usr/bin/env python3
import numpy as np
"""this modual holds function for task 0"""


def one_hot_encode(Y, classes):
    """converts a numeric label vector into a one-hot matrix"""
    m = len(Y)
    onehot = np.zeros((classes, m))
    for i in range(m):
        onehot[i][Y[i]] = 1
    return onehot.T
