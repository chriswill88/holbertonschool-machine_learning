#!/usr/bin/env python3
"""this module contains the function used in task 2"""
import numpy as np


def absorbing(P):
    """This function determinds an absorbing matrix or not"""
    dia = np.diag(P)
    if (dia == 1).all():
        return True

    placehold = np.zeros(dia.shape)
    index = np.argwhere(dia == 1)
    if not len(index):
        return False

    i = 0
    while i < placehold.shape[0] + 1 and not (placehold == 1).all():
        column = P[:, index]
        args = np.where((column > 0) & (column != column[index][0]))
        args = np.unique(args[0])

        if len(args):
            placehold[index] = 1
        index = np.where(placehold == 0)
        if (len(index[0])):
            index = [index[0][0]]
        i += 1
    if (placehold == 1).all():
        return True

    return False
