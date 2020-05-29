#!/usr/bin/env python3
"""this funxtion is used in task 0"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """the cost of a neural network with L2 regularization"""
    w = 0
    for n, value in weights.items():
        if n[0] == 'W':
            w += np.linalg.norm(value)

    return ((cost) + (lambtha/(2 * m)) * w)
