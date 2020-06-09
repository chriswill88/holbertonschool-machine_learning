#!/usr/bin/env python3
"""this modual contains the function pool for task 6"""
import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """
        Performs pooling on images:

        @images
            a numpy.ndarray with shape (m, h, w, c) containing multiple images
            m is the number of images
            h is the height in pixels of the images
            w is the width in pixels of the images
            c is the number of channels in the image

        @kernel_shape
            is a tuple of (kh, kw) containing the kernel shape for the pooling
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

        @stride
            is a tuple of (sh, sw)
            sh is the stride for the height of the image
            sw is the stride for the width of the image

        @mode
            indicates the type of pooling
            max indicates max pooling
            avg indicates average pooling

        Returns: a numpy.ndarray containing the convolved images
    """
    # creating relevant variables
    m = images.shape[0]
    h = images.shape[1]
    w = images.shape[2]
    c = images.shape[3]

    kh = kernel_shape[0]
    kw = kernel_shape[1]

    sh = stride[0]
    sw = stride[1]

    # finding the padding
    ph, pw = 0, 0

    # getting the size of the output
    hi = int((h - kh)/sh) + 1
    wi = int((w - kw)/sw) + 1

    # initializing the output
    new = np.zeros((m, hi, wi, c))

    # doing covolutions
    for row in range(hi):
        for col in range(wi):
            part = images[:, row * sh:(row*sh) + kh, col * sw:(col*sw) + kw]
            if mode == 'max':
                suma = np.max(part, axis=(1, 2))
            elif mode == 'avg':
                suma = part.mean(axis=(1, 2))
            new[:, row, col] = suma
    return new
