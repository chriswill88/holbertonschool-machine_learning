#!/usr/bin/env python3
"""This module contains the class GRUCell"""
import numpy as np


class LSTMCell:
    """
    LSTMCell -
        represents a lstm layer
    """
    def __init__(self, i, h, o):
        # forget gate
        self.Wf = np.random.randn(i + h, h)
        self.bf = np.zeros((1, h))

        # update gate
        self.Wu = np.random.randn(i + h, h)
        self.bu = np.zeros((1, h))

        # intermediate cell state
        self.Wc = np.random.randn(i + h, h)
        self.bc = np.zeros((1, h))

        # output gate
        self.Wo = np.random.randn(i + h, h)
        self.bo = np.zeros((1, h))

        # outputs
        self.Wy = np.random.randn(h, o)
        self.by = np.zeros((1, o))

    def sigmoid(self, x):
        """applies sigmoid operation"""
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        """applies softmax operation"""
        e_x = np.exp(x)
        return e_x / e_x.sum(axis=1, keepdims=True)

    def forward(self, h_prev, c_prev, x_t):
        """
        forward -
            FORWARD PROPOGATION FUNCTION FOR LSTM'S
        """
        con = np.concatenate((h_prev, x_t), axis=1)
        f = self.sigmoid(np.matmul(con, self.Wf) + self.bf)
        i = self.sigmoid(np.matmul(con, self.Wu) + self.bu)
        c_bar = np.tanh(np.matmul(con, self.Wc) + self.bc)

        c = f * c_prev + i * c_bar
        o = self.sigmoid(np.matmul(con, self.Wo) + self.bo)
        h = o * np.tanh(c)

        v = np.matmul(h, self.Wy) + self.by
        y = self.softmax(v)
        return h, c, y
