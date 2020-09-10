#!/usr/bin/python3
import numpy as np


class RNNCell:
    """
    Class RNNCell - this class represents a cell of a simple RNN

        @i: dimensionality of the data
        @h: dimensionality of the hidden states
        @o: dimensionality of outputs
    """

    def __init__(self, i, h, o):
        print("i = {}, h = {}, o = {}".format(i, h, o))

        self.Wh = np.random.randn(i + h, h)
        self.Wy = np.random.randn(h, o)
        self.bh = np.zeros((h))
        self.by = np.zeros((o))

    def forward(self, h_prev, x_t):
        """
        forward - this function implements the forward algorithm
        """
        con = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(con, self.Wh) + self.bh)
        soft = np.matmul(h_next, self.Wy) + self.by

        # softmax application
        y_max = np.max(soft, axis=1, keepdims=True)
        y_exp = np.exp(soft - y_max)
        y = y_exp / y_exp.sum(axis=1, keepdims=True)
        return h_next, y
