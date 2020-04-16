#!/usr/bin/env python3
import copy
"""
    In this modual
    there is the function
    cat_matrixs2D for task 7
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """
        this function concatinates matrixes on a given axis
    """
    newy = copy.deepcopy(mat1)
    if (axis == 0):
        for i in mat2:
            newy.append(i)
    elif (axis == 1):
        for i in range(len(mat2)):
            newy[i].extend(mat2[i])
    return newy
