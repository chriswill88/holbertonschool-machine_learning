#!/usr/bin/env python3
"""performs a same convolution on grayscale images"""
import numpy as np


def convolve_grayscale_same(images, kernel):
    """
    This function same convolves the images

    @images
        is a numpy.ndarray with shape (m, h, w) multiple grayscale images
        m is the number of images
        h is the height in pixels of the images
        w is the width in pixels of the images

    @kernel
        is a numpy.ndarray with shape (kh, kw) the kernel for the convolution
        kh is the height of the kernel
        kw is the width of the kernel

    Returns: a numpy.ndarray containing the convolved images
    """
    # creating relavant variables
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    # calulating padding
    ph = max(int((kh-1)/2), int(kh/2))
    pw = max(int((kw-1)/2), int(kw/2))

    # creating and intializing output
    new = np.zeros((m, h, w))

    # apply padding
    images = np.pad(images, ((0, 0), (ph, ph), (pw, pw)), 'constant')

    # same convolution
    for row in range(h):
        for col in range(w):
            part = images[:, row:row+kh, col:col+kw]
            suma = np.sum(kernel * part, axis=(1, 2))
            new[:, row, col] = suma
    return new
