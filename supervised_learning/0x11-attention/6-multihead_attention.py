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

    def add_heads(self, x, batch):
        """Reconfigures the tensors"""
        x = tf.reshape(x, (batch, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """This function call the multi head attention algorithm"""
        batch = Q.shape[0]
        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = self.add_heads(Q, batch)
        K = self.add_heads(K, batch)
        V = self.add_heads(V, batch)

        output, weight = sdp_attention(Q, K, V, mask)
        scaled_attention = tf.transpose(output, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, num_heads, depth)
        concat_attention = tf.reshape(
            scaled_attention, (batch, -1, self.dm))
        # (batch_size, seq_len_q, d_model)
        outputs = self.linear(concat_attention)
        # (batch_size, seq_len_q, d_model)

        return outputs, weight
