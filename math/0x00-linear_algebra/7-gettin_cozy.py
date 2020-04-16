#!/usr/bin/env python3
"""
    In this modual
    there is the function
    cat_matrixs2D for task 7
"""


def dimensions_getter(var):
    i = 0
    while isinstance(var, list):
        # print("from dim_g var is list and ", var)
        var = var[0]
        i += 1
    return i


def cat_matrices2D(mat1, mat2, axis=0):
    """
        this function concatinates matrixes on a given axis
    """

    if dimensions_getter(mat1) != dimensions_getter(mat2):
        return None
    if (dimensions_getter(mat1) != 2):
        return None

    newy = []
    for i in range(len(mat1)):
        newy.append([])
        for x in mat1[i]:
            newy[i].append(x)

    if (axis == 0):
        for i in mat2:
            newy.append(i)
    elif (axis == 1):
        for i in range(len(mat2)):
            newy[i].extend(mat2[i])
    return newy
