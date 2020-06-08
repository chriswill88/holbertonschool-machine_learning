#!/usr/bin/env python3
"""performs a valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """
        That performs a convolution on grayscale images with custom padding:

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

        Returns: a numpy.ndarray containing the convolved images
    """
    # creating relavant variables
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    ph = padding[0]
    pw = padding[1]

    # applying padding
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    # calculating sizes for output
    hi = h + 2 * ph - kh + 1
    wi = w + 2 * pw - kw + 1

    # intializing output
    new = np.zeros((m, hi, wi))

    # Convolving with custom padding
    for row in range(hi):
        for col in range(wi):
            part = images[:, row:row+kh, col:col+kw]
            suma = np.sum(kernel * part, axis=(1, 2))
            new[:, row, col] = suma
    return new
