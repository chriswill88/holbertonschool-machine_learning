#!/usr/bin/env python3
"""this modual holds function for task 1"""
import numpy as np


def one_hot_decode(one_hot):
    """that converts a one-hot matrix into a vector of labels"""
    if not isinstance(one_hot, np.ndarray) or one_hot.ndim != 2:
        return None

    classes, m = one_hot.shape
    onehot = np.zeros(m, int)

    for i in range(m):
        for x in range(classes):
            if one_hot[x][i] == 1:
                onehot[i] = x
    return onehot
