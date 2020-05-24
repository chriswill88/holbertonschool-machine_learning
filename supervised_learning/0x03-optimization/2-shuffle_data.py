#!/usr/bin/env python3
"""this modual holds the function for task 2"""
import numpy as np


def shuffle_data(X, Y):
    """shuffles two sets of data the same way"""
    r_s = np.random.get_state()
    SX = np.random.permutation(X)
    np.random.set_state(r_s)
    SY = np.random.permutation(Y)
    return SX, SY
