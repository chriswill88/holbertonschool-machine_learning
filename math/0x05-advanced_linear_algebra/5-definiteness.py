#!/usr/bin/env python3
"""This module contains the function for how to solve a determinant"""
import numpy as np


def definiteness(matrix):
    """
    @matrix is a numpy.ndarray of shape (n, n) whose definiteness should be
     calculated

    If matrix is not a numpy.ndarray, raise a TypeError with the message
     matrix must be a numpy.ndarray
    If matrix is not a valid matrix, return None

    Return: the string Positive definite, Positive semi-definite, Negative
     semi-definite, Negative definite, or Indefinite if the matrix is positive
     definite, positive semi-definite, negative semi-definite, negative
     definite of indefinite, respectively
    If matrix does not fit any of the above categories, return None

    You may import numpy as np
    """
    lis = [
        "Positive definite", "Positive semi-definite",
        "Negative semi-definite", "Negative definite", "Indefinite"]

    if not isinstance(matrix, np.ndarray):
        raise TypeError("matrix must be a numpy.ndarray")

    size = matrix.shape[0]
    if len(matrix.shape) == 1 or matrix.shape[0] != matrix.shape[1]:
        return None

    pick = 0
    w, v = np.linalg.eig(matrix)
    for i in w:
        if i < 0:
            pick = 2
    if pick == 0:
        return lis[0]

    for i in range(size):
        mat = matrix[:i + 1, :i + 1]
        d = det(mat, i + 1)
        if d > 0 and pick == 0:
            pick = 0
        elif (i % 2 == 0 and d < 0) or (i % 2 == 1 and d > 0 and pick != 0):
            pick = 3
        elif d == 0 and (pick == 0 or pick == 1):
            pick = 1
        elif d == 0 and (pick == 3 or pick == 2):
            pick = 2
        else:
            pick = -1
    return lis[pick]


