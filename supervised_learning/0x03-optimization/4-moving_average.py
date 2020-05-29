#!/usr/bin/env python3
"""this function is for task 4"""
import numpy as np


def moving_average(data, beta):
    """this function find the moving average"""
    moveave = np.zeros(len(data))
    for i in range(len(data) - 1):
        Vt = (beta * data[i]) + ((1 - beta) * data[i + 1])
        prev = Vt
        moveave[i] = Vt
    return moveave
