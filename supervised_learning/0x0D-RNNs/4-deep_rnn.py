#!/usr/bin/env python3
"""This module contains the function for task 4"""
import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    deep_rnn - performs forward propagation for a deep RNN:

    @rnn_cells is a list of RNNCell instances of length
    l that will be used for the forward propagation
        l - is the number of layers

    @X is the data to be used, given as a numpy.ndarray
    of shape (t, m, i)
        t - is the maximum number of time steps
        m - is the batch size
        i - is the dimensionality of the data

    @h_0 is the initial hidden state, given as a numpy.ndarray
    of shape (l, m, h)
        h - is the dimensionality of the hidden state

    Returns: H, Y
    H is a numpy.ndarray containing all of the hidden states
    Y is a numpy.ndarray containing all of the outputs
    """
    layers = len(rnn_cells)
    time, m, i = X.shape
    _, _, h = h_0.shape

    H = np.zeros((time + 1, layers, m, h))
    H[0] = h_0

    for t in range(time):
        for lay, layer in enumerate(rnn_cells):
            if lay == 0:
                h, y = layer.forward(H[t, lay], X[t])
            else:
                h, y = layer.forward(H[t, lay], h)
            H[t + 1, lay] = h

            if lay == layers - 1:
                if t == 0:
                    Y = y
                else:
                    Y = np.concatenate((Y, y))

    return H, Y.reshape(time, m, Y.shape[-1])
