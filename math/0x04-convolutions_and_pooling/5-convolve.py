#!/usr/bin/env python3
"""this modual contains the function convolve_channels for task 5"""
import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """
        performs a convolution on multi channel images:

        @images
            a numpy.ndarray with shape (m, h, w, c) containing multiple images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
            c is the number of channels in the image

        @kernels
            a numpy.ndarray (kh, kw, c, nc) containing kernels for convolution
            kh is the height of a kernel
            kw is the width of a kernel
            nc is the number of kernels

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
    # creating relevant variables
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]

    kh = kernels.shape[0]
    kw = kernels.shape[1]
    nc = kernels.shape[3]

    sh = stride[0]
    sw = stride[1]

    # finding the padding
    if isinstance(padding, tuple):
        ph = padding[0]
        pw = padding[1]

    elif padding is 'same':
        ph = int(((h - 1) * sh + kh - h)/2) + 1
        pw = int(((w - 1) * sw + kw - w)/2) + 1

    elif padding == 'valid':
        ph, pw = 0, 0

    # getting the size of the output
    hi = int((h + 2*ph - kh)/sh) + 1
    wi = int((w + 2*pw - kw)/sw) + 1

    # initializing the output
    new = np.zeros((m, hi, wi, nc))

    # applying padding
    newimage = np.pad(images, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    # doing covolutions
    for row in range(hi):
        for col in range(wi):
            for k in range(nc):
                part = newimage[:, row*sh:(row*sh)+kh, col*sw:(col*sw)+kw]
                suma = np.sum(kernels[:, :, :, k] * part, axis=(1, 2, 3))
                new[:, row, col, k] += suma

    return new
