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
        print("custom")
        ph = padding[0]
        pw = padding[1]

        hi = int(h + 2 * ph - kh/sh) + 1
        wi = int(w + 2 * pw - kw/sw) + 1

        mh = hi = h + 2 * ph - kh + 1
        mw = wi = w + 2 * pw - kw + 1

    elif padding is 'same':
        ph = max(int((kh-1)/2), int(kh/2))
        pw = max(int((kw-1)/2), int(kw/2))

        hi = mh = h
        wi = mw = w

    elif padding == 'valid':
        ph, pw = 0, 0

        hi = int((h-kh)/sh) + 1
        wi = int((w-kw)/sw) + 1

        mh = h-kh+1
        mw = w-kw+1

    new = np.zeros((m, hi, wi))
    newimage = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    r = 0
    for row in range(0, mh, sh):
        c = 0
        for col in range(0, mw, sw):
            part = newimage[:, row:row+kh, col:col+kw]
            suma = np.sum(kernel * part, axis=(1, 2))
            new[:, r, c] = suma
            c += 1
        r += 1
    return new
