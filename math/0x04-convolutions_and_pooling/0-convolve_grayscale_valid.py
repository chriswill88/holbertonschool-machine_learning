#!/usr/bin/env python3
"""performs a valid convolution on grayscale images"""
import numpy as np


def convolve_grayscale_valid(images, kernel):
    """
    This function validly convolves the images

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
    # creating important variables
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]

    kh = kernel.shape[0]
    kw = kernel.shape[1]

    # getting the size of output
    nh, nw = h-kh+1, w-kw+1

    # initializing the output
    new = np.zeros((m, nh, nw))

    # Convolulving validly
    for row in range(nh):
        for col in range(nw):
            part = images[:, row:row+kh, col:col+kw]
            suma = np.sum(kernel * part, axis=(1, 2))
            new[:, row, col] = suma

    return new
