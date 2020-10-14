#!/usr/bin/env python3
"""This module contains the function for task 5"""
import numpy as np


class BidirectionalCell:
    """
    BidirectionalCell:
        This class represents a bidirectional cell of an RNN
    """
    def __init__(self, i, h, o):
        """
        @i: is the dimensionality of the data
        @h: is the dimensionality of the hidden states
        @o: is the dimensionality of the outputs

        Creates the public instance attributes Whf, Whb, Wy, bhf, bhb, by that
         represent the weights and biases of the cell
            @Whf: and bhfare for the hidden states in the forward direction
            @Whb: and bhbare for the hidden states in the backward direction
            @Wy: and byare for the outputs
        """
        self.Whf = np.random.randn(i + h, h)
        self.bhf = np.zeros((1, h))

        self.Whb = np.random.randn(i + h, h)
        self.bhb = np.zeros((1, h))

        self.Wy = np.random.randn(h + i + o, o)
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        forward - this function implements the forward algorithm

        @x_t is a numpy.ndarray of shape (m, i) that contains the data input
        for the cell
            m is the batche size for the data

        @h_prev is a numpy.ndarray of shape (m, h) containing the
        previous hidden state

        The output of the cell should use a softmax activation function
        Returns: h_next, y
            h_next is the next hidden state
            y is the output of the cell
        """
        con = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.matmul(con, self.Whf) + self.bhf)
        return h_next

    def backward(self, h_next, x_t):
        """
        backwards - this function implements the backwards algorithm

        calculates the hidden state in the backward direction for one time step
        """
        con = np.concatenate((h_next, x_t), axis=1)
        h_prev = np.tanh(np.matmul(con, self.Whb) + self.bhb)
        return h_prev

    def output(self, H):
        """This function calculates and computes the outputs"""
        t, m, h = H.shape
        h = int(h / 2)
        Y = np.zeros((t, m, 5))
        for i in range(t):
            soft = np.matmul(H[i - 1], self.Wy) + self.by

            # softmax application
            y_max = np.max(soft, axis=1, keepdims=True)
            y_exp = np.exp(soft - y_max)
            y = y_exp / y_exp.sum(axis=1, keepdims=True)
            Y[t - i - 1] = y
        return Y
