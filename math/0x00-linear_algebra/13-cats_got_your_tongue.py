#!/usr/bin/env python3
"""this modual hold the functions for task 13"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """this function returns a transposed matrix"""
    return np.concatenate((mat1, mat2), axis).copy()
