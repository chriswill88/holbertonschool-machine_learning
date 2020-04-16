#!/usr/bin/env python3
"""
    In this modual
    there is the function
    add_matrices2d for task 5
"""


def add_matrices2D(mat1, mat2):
    """
        add_matrices2D - does element wise addition for matrices
    """
    if (len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0])):
        return None

    new = []
    for i in range(len(mat1)):
        new.append([])
        for x in range(len(mat1[0])):
            new[i].append(mat1[i][x] + mat2[i][x])

    return new
