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
        super(MultiHeadAttention, self).__init__()

        self.h = h
        self.dm = dm
        self.depth = int(dm / h)
        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """
        Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is
         (batch_size, num_heads, seq_len, depth)
        """
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """This function call the multi head attention algorithm"""
        batch_size = tf.shape(Q)[0]

        Q = self.Wq(Q)
        # (batch_size, seq_len, d_model)
        K = self.Wk(K)
        # (batch_size, seq_len, d_model)
        V = self.Wv(V)
        # (batch_size, seq_len, d_model)

        Q = self.split_heads(Q, batch_size)
        # (batch_size, num_heads, seq_len_q, depth)
        K = self.split_heads(K, batch_size)
        # (batch_size, num_heads, seq_len_k, depth)
        V = self.split_heads(V, batch_size)
        # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = sdp_attention(
            Q, K, V, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention, (
            batch_size, -1, self.dm))
        # (batch_size, seq_len_q, d_model)

        output = self.linear(concat_attention)
        # (batch_size, seq_len_q, d_model)

        return output, attention_weights
