#!/usr/bin/env python3
"""
    In this modual
    there is the function
    add_matrices2d for task 5
"""
shape = __import__('2-size_me_please').matrix_shape


def add_matrices2D(mat1, mat2):
    """
        add_matrices2D - does element wise addition for matrices
    """
    if (shape(mat1) != shape(mat2)):
        return None

    new = []
    for i in range(len(mat1)):
        new.append([])
        for x in range(len(mat1[0])):
            new[i].append(mat1[i][x] + mat2[i][x])
    return new
