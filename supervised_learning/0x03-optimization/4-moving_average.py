#!/usr/bin/env python3
"""this function is for task 4"""
import numpy as np


def moving_average(data, beta):
    """this function find the moving average"""
    moveave = np.zeros(len(data))
    prev = data[0]
    for i in range(len(data)):
        Vt = (beta * prev) + ((1 - beta) * data[i])
        prev = Vt
        moveave[i] = Vt
    return moveave
