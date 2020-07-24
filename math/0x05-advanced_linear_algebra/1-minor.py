#!/usr/bin/env python3
"""This module contains the function for how to solve a determinant"""


def check(matrix):
    """
    The error checker
    returns size of square
    """
    size = len(matrix)

    if not isinstance(matrix, list) or len(matrix) == 0:
        raise TypeError("matrix must be a list of lists")

    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
        if len(row) != size and size != 1 or not len(row):
            raise ValueError("matrix must be a non-empty square matrix")
    return size


def submatrix(matrix, h, w):
    """This function gets the submatrix"""
    sub = []
    idx = -1
    for ind, i in enumerate(matrix):
        if ind != h:
            idx += 1
            sub.append([])
            for x in range(len(i)):
                if x != w and ind != h:
                    sub[idx].append(matrix[ind][x])
    return sub


def det(matrix, size):
    """computes the determinant"""
    sum = []
    if size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        for x in range(size):
            sum.append([])
            for i in range(size):
                sum[x].append(det(submatrix(matrix, x, i), size - 1))
    return sum


def minor(matrix):
    """
    This function solves a determinant
    @matrix is a list of lists whose determinant should be calculated

    If matrix is not a list of lists, raise a TypeError with the message
     matrix must be a list of lists
    If matrix is not square, raise a ValueError with the message matrix
     must be a square matrix
    The list [[]] represents a 0x0 matrix

    Returns: the determinant of matrix
    """
    # check function
    size = check(matrix)

    # determinant
    if size == 1:
        return 1
    if size == 2:
        return [sub[::-1] for sub in matrix[::-1]]
    return det(matrix, size)
