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
    sw = stride[1]

    if isinstance(padding, tuple):
        ph = padding[0]
        pw = padding[1]

    elif padding is 'same':
        ph = int(((h - 1) * sh + kh - h)/2) + 1
        pw = int(((w - 1) * sw + kw - w)/2) + 1

    elif padding == 'valid':
        ph, pw = 0, 0

    hi = int((h + 2*ph - kh)/sh) + 1
    wi = int((w + 2*pw - kw)/sw) + 1

    new = np.zeros((m, hi, wi))
    newimage = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    for row in range(hi):
        for col in range(wi):
            part = newimage[:, row * sh:(row*sh) + kh, col * sw:(col*sw) + kw]
            suma = np.sum(kernel * part, axis=(1, 2))
            new[:, row, col] = suma
    return new
