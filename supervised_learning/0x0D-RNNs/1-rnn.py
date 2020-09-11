#!/usr/bin/env python3
"""This module contains the function rnn"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    rnn - performs forward propagation for a simple RNN
        @rnn_cell: an instance of RNNCell that will be used for the
         forward propagation
        @X is the data to be used
         given as a numpy.ndarray of shape (t, m, i)
            - t is the maximum number of time steps
            - m is the batch size
            - i is the dimensionality of the data
        @h_0: the initial hidden state
         given as a numpy.ndarray of shape (m, h)
            - h is the dimensionality of the hidden state

    Returns:
        H is a numpy.ndarray containing all of the hidden states
        Y is a numpy.ndarray containing all of the outputs
    """
    t, m, i = X.shape
    _, h = h_0.shape
    hold_h = np.expand_dims(h_0, 0)
    hold_y = None

    for o in range(t):
        inp = X[o]
        h_0, y = rnn_cell.forward(h_0, inp)
        y = np.expand_dims(y, 0)
        h_s = np.expand_dims(h_0, 0)

        hold_h = np.append(hold_h, h_s, 0)
        hold_y = np.append(hold_y, y, 0) if hold_y is not None else y

    return hold_h, hold_y
