#!/usr/bin/env python3
"""
    In this modual
    there is the function
    cat_matrixs2D for task 7
"""


def dimensions_getter(var):
    """returns the number of dimensions"""
    i = 0
    lis = []
    while isinstance(var, list):
        lis.append(len(var))
        var = var[0]
        i += 1
    return list(reversed(lis))


def cat_matrices2D(mat1, mat2, axis=0):
    """this function concatinates matrixes on a given axis"""

    shape1 = dimensions_getter(mat1)
    shape2 = dimensions_getter(mat2)

    if (len(shape1) != 2 or len(shape2) != 2):
        return (None)
    try:
        if (shape1[axis] != shape2[axis]):
            return None
    except IndexError:
        return None

    new = []
    for i in range(len(mat1)):
        new.append([])
        for x in mat1[i]:
            new[i].append(x)

    if (axis == 0):
        for i in mat2:
            new.append(i)
    elif (axis == 1):
        for i in range(len(mat2)):
            new[i].extend(mat2[i])
    return new
