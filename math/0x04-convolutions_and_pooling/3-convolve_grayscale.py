#!/usr/bin/env python3
"""performs strided convolutions on grayscale images"""
import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """
        That performs strided convolution on grayscale images:

        @images
            a numpy.ndarray with shape (m, h, w, c) containing multiple images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
            c is the number of channels in the image

        @kernel
            a numpy.ndarray with shape (kh, kw, c) the kernel for convolution
            kh is the height of the kernel
            kw is the width of the kernel

        @padding
            either a tuple of (ph, pw), ‘same’, or ‘valid’
            if ‘same’, performs a same convolution
            if ‘valid’, performs a valid convolution
            if a tuple:
            ph is the padding for the height of the image
            pw is the padding for the width of the image
            the image should be padded with 0’s

        @stride is a tuple of (sh, sw)
            sh is the stride for the height of the image
            sw is the stride for the width of the image

        Returns: a numpy.ndarray containing the convolved images
    """
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
