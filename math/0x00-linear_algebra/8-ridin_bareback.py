#!/usr/bin/env python3
"""this modual hold the functions for task 8"""


def dimensions_getter(var):
    """returns the shape"""
    i = 0
    lis = []
    while isinstance(var, list):
        lis.append(len(var))
        var = var[0]
        i += 1
    return (lis)


def mat_mul(mat1, mat2):
    """multiplies two matrices"""
    shape1 = dimensions_getter(mat1)
    shape2 = dimensions_getter(mat2)

    height = shape1[0]
    width = shape2[-1]

    if (shape1[-1] != shape2[0]):
        return None

    length = shape1[-1]
    new = []

    for y in range(len(mat1)):
        new.append([])
        for x in range(len(mat2[0])):
            sum = 0
            for z in range(length):
                sum += mat1[y][z] * mat2[z][x]
            new[y].append(sum)
    return(new)