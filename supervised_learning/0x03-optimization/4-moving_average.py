#!/usr/bin/env python3
"""this function is for task 4"""
import numpy as np


def moving_average(data, beta):
    """this function find the moving average"""
    moveave = np.zeros(len(data))
    moveave[0] = data[0]
    prev = data[0]
    for i in range(len(data) - 1):
        print(prev)
        Vt = (beta * prev) + ((1 - beta) * data[i + 1])
        prev = Vt
        moveave[i + 1] = Vt
    return moveave
