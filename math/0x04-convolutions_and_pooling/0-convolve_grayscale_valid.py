#!/usr/bin/env python3
"""performs a valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    kh = kernel.shape[0]
    kw = kernel.shape[1]
    new = np.zeros((m, h-kh+1, w-kw+1))

    for row in range(h):
        for col in range(w):
            if col+kh < h and row+kw < w:
                part = images[:, row:row+kh, col:col+kw]
                suma = np.sum(kernel * part, axis=(1, 2))
                new[:, row, col] = suma

    return new
