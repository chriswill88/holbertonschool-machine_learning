#!/usr/bin/env python3
"""this module contains a function for task 5"""
import tensorflow as tf
sdp_attention = __import__('5-sdp_attention').sdp_attention


class MultiHeadAttention(tf.keras.layers.Layer):
    """
        MultiHeadAttention:
            inherits from tensorflow.keras.layers.Layer to perform multi head
             attention:
    """
    def __init__(self, dm, h):
        """
        @h - the number of heads
        @dm - the dimensionality of the model
        @depth - the depth of each attention head
        @Wq - a Dense layer with dm units, used to generate the query matrix
        @Wk - a Dense layer with dm units, used to generate the key matrix
        @Wv - a Dense layer with dm units, used to generate the value matrix
        @linear - a Dense layer with dm units, used to generate the attention
         output
        """
        self.h = h
        self.dm = dm
        self.depth = int(dm / h)
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def __call__(self, Q, K, V, mask):
        """This function call the multi head attention algorithm"""
        batch = Q.shape[0]

        for i in range(batch):
            for x in range(self.h):
                output, weight = sdp_attention(Q[i], K[i], V[i], mask)
                output = tf.expand_dims(output, axis=0)
                weight = tf.expand_dims(weight, axis=0)
                if x == 0:
                    headW = weight
                else:
                    headW = tf.concat((headW, weight), axis=0)
            headW = tf.expand_dims(headW, axis=0)

            if i != 0:
                weights = tf.concat((weights, headW), axis=0)
                outputs = tf.concat((outputs, output), axis=0)
            else:
                weights = headW
                outputs = output
        return outputs, weights
