#!/usr/bin/env python3
"""this modual contains the function for task 2"""
import numpy as np


def variance(X, C):
    """calculates the total for a data set"""
    try:
        n, d = X.shape
        k = C.shape[0]

        lis = ((X - np.split(C, k))**2).sum(2)
        lis = np.amin(lis, 0).sum()

        return lis
    except Exception:
        return None
