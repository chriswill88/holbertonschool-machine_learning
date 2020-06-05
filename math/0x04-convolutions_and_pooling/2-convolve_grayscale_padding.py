#!/usr/bin/env python3
"""performs a valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """that performs a convolution on grayscale images with custom padding:"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    ph = padding[0]
    pw = padding[1]

    hi = h - 2*ph - kh + 1
    wi = w - 2*pw - kh + 1
    new = np.zeros((m, h, w))
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    for row in range(hi):
        for col in range(wi):
            part = images[:, row:row+kh, col:col+kw]
            suma = np.sum(kernel * part, axis=(1, 2))
            new[:, row, col] = suma
    return new
