#!/usr/bin/env python3
"""performs a valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """this function validly convolves the images"""
    m = images.shape[0]
    n = images.shape[1]
    f = kernel.shape[1]
    new = np.zeros((m, n-f+1, n-f+1))

    for row in range(n):
        for col in range(n):
            if col+3 >= n:
                break
            if row+3 >= n:
                break
            part = images[:, row:row+3, col:col+3]
            suma = np.sum(kernel * part, axis=(1, 2))
            new[:, row, col] = suma
    return new
