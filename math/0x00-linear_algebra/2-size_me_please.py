#!/usr/bin/env python3
"""
    In this modual the function for
    task 2, matrix_shape is created
"""


def matrix_shape(matrix):
    """
        matrix_shape: this function
        returns the shape of any matrix
    """
    lis = []
    return matrixy(matrix, lis)


def matrixy(matrix, lis):
    """
        matrixy: the helper function for matrix_shape
        matrixy figures out the shape using recursion
    """
    if (isinstance(matrix, list)):
        lis.append(len(matrix))
        matrixy(matrix[0], lis)
    return (lis)
