#!/usr/bin/env python3
"""This module contains the function for how to solve a determinant"""


def check(matrix):
    """
    The error checker
    returns size of square
    """
    size = len(matrix)

    if not isinstance(matrix, list) or size == 0:
        raise TypeError("matrix must be a list of lists")

    for row in matrix:
        if not isinstance(row, list):
            raise TypeError("matrix must be a list of lists")
        if len(row) != size and size != 1 or not len(row):
            raise ValueError("matrix must be a non-empty square matrix")
    return size


def submatrix(matrix, index):
    """This function gets the submatrix"""
    size = len(matrix)
    matrix = matrix[1:]

    return [[matrix[x][i] for i in range(
           size) if i != index] for x in range(size - 1)]


def minOfMatrix(matrix, h, w):
    """This function creates a list of list of minors from indices"""
    size = len(matrix)
    return [[matrix[x][i] for i in range(
            size) if i != h] for x in range(size) if x != w]


def det(matrix, size):
    """computes the determinant"""
    sum = 0
    mul = 1
    if size == 1:
        return matrix[0][0]
    if size == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    else:
        for i in range(size):
            sum += mul * (matrix[0][i] * det(submatrix(matrix, i), size - 1))
            mul *= -1
    return sum


def inverse(matrix):
    """
    This function calculates the adjugate
    @matrix is a list of lists whose determinant should be calculated

    If matrix is not a list of lists, raise a TypeError with the message
     matrix must be a list of lists
    If matrix is not square or is empty, raise a ValueError
     with the message matrix must be a non-empty square matrix
    The list [[]] represents a 0x0 matrix

    Returns: the adjugate of matrix
    """
    # check function
    size = check(matrix)
    d = det(matrix, size)
    if d == 0:
        return None
    sub = []

    # matrix of minors
    if size == 1:
        return [[1/d]]
    if size == 2:
        sub = [sub[::-1] for sub in matrix[::-1]]
        sub[0][1] *= -1
        sub[1][0] *= -1
        sub[0][1], sub[1][0] = sub[1][0], sub[0][1]
        return [[x/d for x in i] for i in sub]

    else:
        sub = [[det(minOfMatrix(matrix, x, i), size - 1)
                for i in range(size)] for x in range(size)]
    # cofactors
    mul = 1
    for i in range(len(sub)):
        for x in range(len(sub)):
            sub[i][x] *= mul
            mul *= -1

    # adjugate
    size = len(sub)
    adj = [[sub[x][i] for x in range(size)] for i in range(size)]

    # inverse
    d = det(matrix, size)
    print(adj)
    print([[x for x in i] for i in adj])
