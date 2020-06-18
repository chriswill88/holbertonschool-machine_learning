#!/usr/bin/env python3
"""This modual contains the function used in task 3"""
import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    performs back propagation over a pooling layer of a neural network:

    @dA
        a numpy.ndarray of shape
            (m, h_new, w_new, c_new) containing
            the partial derivatives with respect
            to the output of the pooling layer
        m is the number of examples
        h_new is the height of the output
        w_new is the width of the output
        c_new is the number of channels in the output`

    @A_prev
        a numpy.ndarray of shape (m, h_prev, w_prev, c_prev) the output of
            the previous layer
        h_prev is the height of the previous layer
        w_prev is the width of the previous layer
        c_prev is the number of channels in the previous layer
    @kernel_shape
        a tuple of (kh, kw) size of the kernel for the pooling
        kh is the kernel height
        kw is the kernel width
    @stride
        is a tuple of (sh, sw) containing the strides for the pooling
        sh is the stride for the height
        sw is the stride for the width
    @mode
        a string containing either max or avg | indicating the type of pooling
    """
    m, h_new, w_new, c_n = dA.shape
    _, h_prev, w_prev, _ = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    # initializing the output
    new = np.zeros(A_prev.shape)

    # doing pooling
    for i in range(m):
        for r in range(h_new):
            for c in range(w_new):
                for k in range(c_n):
                    # iterate through all the values in the derivative of z
                    dA_cust = dA[i, r, c, k]
                    aP_cust = A_prev[i, r*sh: r*sh + kh, c*sw:c*sw + kw, k]
                    # to get the derivative of A we have to multiply the value
                    # of dA to the channel of W
                    if mode == 'max':
                        mask = np.zeros(kernel_shape)
                        ma_ap = np.max(aP_cust)
                        mask = np.where(aP_cust == ma_ap, 1, 0)
                        new[i, r*sh: r*sh + kh, c*sw:c*sw + kw, k] += \
                            dA_cust * mask

                    elif mode == 'avg':
                        mask = np.ones(kernel_shape)/(kh * kw)
                        mask *= dA_cust
                        new[i, r*sh: r*sh + kh, c*sw:c*sw + kw, k] += mask
    return new
