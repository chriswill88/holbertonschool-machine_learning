#!/usr/bin/env python3
"""performs a valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    m = images.shape[0]
    n1 = images.shape[1]
    n2 = images.shape[2]
    f1 = kernel.shape[0]
    f2 = kernel.shape[1]

    new = np.zeros((m, n1-f1+1, n2-f2+1))

    for row in range(n1):
        for col in range(n2):
            if col+f2 < n2 and row+f1 < n1:
                part = images[:, row:row+f1, col:col+f2]
                suma = np.sum(kernel * part, axis=(1, 2))
                new[:, row, col] = suma

    return new
