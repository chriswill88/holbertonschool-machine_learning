#!/usr/bin/env python3
"""this modual holds function for task 0"""
import numpy as np


def one_hot_encode(Y, classes):
    """converts a numeric label vector into a one-hot matrix"""
    m = len(Y)
    if isinstance(classes, int):
        return None
    if classes < np.amax(Y):
        return None
    onehot = np.zeros((classes, m))
    for i in range(m):
        onehot[Y[i]][i] = 1
    return onehot
