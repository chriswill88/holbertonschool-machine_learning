#!/usr/bin/env python3
"""This modual contains the function used in task 2"""
import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Performs back propagation over a convolutional layer:

    @dZ
        a numpy.ndarray of shape (m, h_new, w_new, c_new) containing
            the partial derivatives with respect to the unactivated output
            of the convolutional layer
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

    @W
        a numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the kernels
        kh is the filter height
        kw is the filter width

    @b
        a numpy.ndarray of shape (1, 1, 1, c_new) containing the biases

    @padding
        a string that is either same or valid, indicatin type of padding used

    @stride
        a tuple of (sh, sw) containing the strides for the convolution
        sh is the stride for the height
        sw is the stride for the width

    """
    # Retrieving information
    m, h_new, w_new, c_new = dZ.shape
    h_m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, c_prev, c_new = W.shape
    sh, sw = stride

    # setting padding
    ph, pw = 0, 0

    if padding == 'same':
        ph = int(np.ceil((h_prev - 1) * sh + kh - h_prev)/2)
        pw = int(np.ceil((w_prev - 1) * sw + kw - w_prev)/2)

    # applying padding
    A_prev = np.pad(A_prev, ((0, 0), (ph, ph), (pw, pw), (0, 0)), 'constant')

    # initializing the output
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)

    # doing covolutions
    for i in range(m):
        for r in range(h_new):
            for c in range(w_new):
                for k in range(c_new):
                    # iterate through all the values in the derivative of z
                    dZ_cust = dZ[i, r, c, k]
                    W_cust = W[:, :, :, k]
                    # to get the derivative of A we have to multiply the value
                    # of dZ to the channel of W
                    dA_prev[i, r*sh:r*sh+kh, c*sw:c*sw+kw, :] += W_cust*dZ_cust

                    # To get the derivative of W we have to multiply to
                    # value of dz by value a_prev accross all channels
                    dW[:, :, :, k] += A_prev[i, r*sh:r*sh+kh, c*sw:c*sw+kw, :]\
                        * dZ_cust

    # getting derivative of b is taking the sum of dZ
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # remove padding
    dA_prev = dA_prev[:, ph:dA_prev.shape[1]-ph, pw:dA_prev.shape[2]-pw, :]
    return dA_prev, dW, db
