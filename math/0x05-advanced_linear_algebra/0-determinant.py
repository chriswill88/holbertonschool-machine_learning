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
        if len(row) != size and size != 1:
            raise ValueError("matrix must be a square matrix")
    return size


def submatrix(matrix, index):
    """This function gets the submatrix"""
    sub = []
    # print("step 1: retrieve\n", matrix)
    matrix = matrix[1:]
    # print("step 2: cut off top\n", matrix)

    for ind, i in enumerate(matrix):
        sub.append([])
        for x in range(len(i)):
            if x != index:
                sub[ind].append(matrix[ind][x])
    return sub


def det(matrix, size):
    """computes the determinant"""
    comp_size = size - 2
    sum = 0
    mul = 1
    if size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        for i in range(size):
            sum += mul * (matrix[0][i] * det(submatrix(matrix, i), size - 1))
            mul *= -1
    return sum


def determinant(matrix):
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
        if len(matrix[0]) == 0:
            return 1
        else:
            return matrix[0][0]
    return det(matrix, size)
