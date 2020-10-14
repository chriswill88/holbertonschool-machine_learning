#!/usr/bin/env python3
"""This module contains the function for task 8"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
        This function computes forward propogation for task 8
    """
    t, m, i = X.shape
    _, h = h_0.shape
    nh = h*2

    H = np.zeros((t, m, 2 * h))
    h_n = h_0
    h_o = h_t

    for i in range(t):
        h_n = bi_cell.forward(X[i], h_n)
        h_o = bi_cell.backward(X[t - 1 - i], h_o)
        n = np.concatenate((h_o, h_n), 1)
        H[i] = n

    Y = bi_cell.output(H)

    return H, Y
