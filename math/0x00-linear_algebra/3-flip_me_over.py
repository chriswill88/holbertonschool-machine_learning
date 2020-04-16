#!/usr/bin/env python3
"""
    In this modual there is the
    matrix_transpose function
    for task 3.
"""


def matrix_transpose(matrix):
    """
        This function transposes a matrix.
    """
    lis = []
    for i in range(len(matrix[0])):
        lis.append([])
        for x in matrix:
            lis[i].append(x[i])
    return lis
