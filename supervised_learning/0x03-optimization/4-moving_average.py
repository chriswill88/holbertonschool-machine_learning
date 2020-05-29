#!/usr/bin/env python3
"""this function is for task 4"""


def moving_average(data, beta):
    """this function find the moving average"""
    moveave = []
    con = 1 - beta
    prev = 0
    for i in range(len(data)):
        part = (beta * prev) + (con * data[i])
        Vt = part/(1 - beta**(i + 1))
        prev = part
        moveave += [Vt]

    return moveave
