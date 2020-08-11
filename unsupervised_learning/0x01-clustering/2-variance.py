#!/usr/bin/env python3
"""this modual contains the function for task 2"""
import numpy as np


def variance(X, C):
    """calculates the total for a data set"""
    try:
    n, d = X.shape
    k = C.shape[0]
    hold = np.zeros((k, n))
    lis = np.zeros((n))
    print(X - C)
    for j in range(k):
        for i in range(n):
            var = ((X[i] - C[j])**2).sum(0)
            if lis[i] == 0 or var < lis[i]:
                lis[i] = var

    lis = np.amin(lis.sum())

    return lis
