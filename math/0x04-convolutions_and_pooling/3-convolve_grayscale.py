#!/usr/bin/env python3
"""performs a valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """that performs a convolution on grayscale images with custom padding:"""
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    sh = stride[0]
    sw = stride[0]

    if isinstance(padding, tuple):
        ph = padding[0]
        pw = padding[1]

    elif padding == 'same':
        ph = int(max((kh-1)/2), (kh/2))
        pw = int(max((kw-1)/2), (kw/2))

    elif padding == 'valid':
        ph, pw = 0, 0

    hi = np.floor_divide(h + 2 * ph - kh, sh) + 1
    wi = np.floor_divide(w + 2 * pw - kw, sw) + 1

    new = np.zeros((m, hi, wi))
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    mh = h + 2 * ph - kh + 1
    mw = w + 2 * pw - kw + 1

    r = 0
    for row in range(0, mh, sh):
        c = 0
        for col in range(0, mw, sw):
            part = images[:, row:row+kh, col:col+kw]
            suma = np.sum(kernel * part, axis=(1, 2))
            new[:, r, c] = suma
            c += 1
        r += 1
    return new
