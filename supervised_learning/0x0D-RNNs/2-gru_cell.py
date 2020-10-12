#!/usr/bin/env python3
"""This module contains the class GRUCell"""
import numpy as np


class GRUCell:
    """
    GRUCell -
        a class that represents a GRU layer
    @i is the dimensionality of the data
    @h is the dimensionality of the hidden state
    @o is the dimensionality of the outputs

    Creates the public instance attributes Wz, Wr, Wh, Wy, bz, br, bh, by that
    represent the weights and biases of the cell
        @Wz and bz are for the update gate
        @Wr and br are for the reset gate
        @Wh and bh are for the intermediate hidden state
        @Wy and by are for the output
    """
    def __init__(self, i, h, o):
        # update gate
        self.Wz = np.random.randn(i + h, h)
        self.bz = np.zeros((1, h))

        # reset gate
        self.Wr = np.random.randn(i + h, h)
        self.br = np.zeros((1, h))

        # intermediate hidden state
        self.Wh = np.random.randn(i + h, h)
        self.bh = np.zeros((1, h))

        # output
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """applies sigmoid operation"""
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """applies softmax operation"""
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """forward algorithm for gru's"""
        con = np.concatenate((h_prev, x_t), axis=1)
        # update and forget gates
        z = self.sigmoid(np.matmul(con, self.Wz) + self.bz)
        r = self.sigmoid(np.matmul(con, self.Wr) + self.br)

        # current state
        con = np.concatenate(((r * h_prev), x_t), axis=1)
        h = np.tanh(np.matmul(con, self.Wh) + self.bh)

        # layer output
        s = (1 - z) * h_prev + z * h

        # softmax layer
        y = self.softmax(np.matmul(s, self.Wy) + self.by)
        return s, y
