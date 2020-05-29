#!/usr/bin/env python3
"""this funxtion is used in task 0"""
import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """the cost of a neural network with L2 regularization"""
    summ = 0
    for n, value in weights.items():
        if n[0] == "W":
            summ += np.sum(value**2)

    return (cost) + lambtha/(2 * m) * summ**(1/2)
