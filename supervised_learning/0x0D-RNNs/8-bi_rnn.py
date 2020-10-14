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

    h_o = [h_0]
    h_i = [h_t]
    for i in range(t):
        h_0 = bi_cell.forward(h_0, X[i])
        h_t = bi_cell.backward(h_t, X[t - 1 - i])

        h_o.append(h_0)
        h_i.append(h_t)
    h_o = np.array(h_o)
    h_i = np.array([i for i in reversed(h_i)])
    H = np.concatenate((h_o[1:], h_i[:-1]), -1)
    Y = bi_cell.output(H)

    return H, Y
