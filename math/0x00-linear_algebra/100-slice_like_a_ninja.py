#!/usr/bin/env python3
"""this modual contains the function for task 100"""


def np_slice(matrix, axes={}):
    """slices np arrays and returns them"""
    start, stop, end = None, None, None

    ax = len(matrix)
    for a, slic in axes.items():
        try:
            start = slic[0] if 0 < ax else None
        except IndexError:
            pass

        try:
            stop = slic[1] if 1 < ax else None
        except IndexError:
            pass
        try:
            end = slic[2] if 2 < ax else None
        except IndexError:
            pass
        slc = [slice(None)] * len(matrix.shape)
        slc[a] = slice(start, stop, end)

        matrix = matrix[slc]
        return matrix
