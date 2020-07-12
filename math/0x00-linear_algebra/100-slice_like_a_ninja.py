#!/usr/bin/env python3
"""this modual contains the function for task 100"""


def np_slice(matrix, axes={}):
    """slices np arrays and returns them"""
    start, stop, end = None, None, None
    slc = [slice(None, None, None)] * len(matrix.shape)

    for a, slic in axes.items():
        slc[a] = slice(*slic)

    return matrix[slc].copy()
