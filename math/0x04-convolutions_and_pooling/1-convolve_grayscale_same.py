#!/usr/bin/env python3
"""performs a same convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """this function same convolves the images"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    Ph = int((kh-1)/2)
    Pw = int((kw-1)/2)

    hi = h + 2*Ph - kh + 1
    wi = w + 2*Pw - kw + 1

    new = np.zeros((m, hi, wi))
    images = np.pad(images, ((0,), (Ph,), (Pw,)), 'constant')

    for row in range(hi):
        for col in range(wi):
            part = images[:, row:row+kh, col:col+kw]
            suma = np.sum(kernel * part, axis=(1, 2))
            new[:, row, col] = suma
    return new
