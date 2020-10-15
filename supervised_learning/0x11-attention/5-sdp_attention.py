#!/usr/bin/env python3
"""this module contains a function for task 4"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    sdp_attention -
        this function calculates the scaled dot product attention
        @Q: is a tensor with its last two dimensions as (..., seq_len_q, dk)
         containing the query matrix
        @K: is a tensor with its last two dimensions as (..., seq_len_v, dk)
         containing the key matrix
        @V: is a tensor with its last two dimensions as (..., seq_len_v, dv)
         containing the value matrix
        @mask: is a tensor that can be broadcast into (..., seq_len_q,
         seq_len_v) containing the optional mask, or defaulted to None
    Returns: output, weights
        @output a tensor with its last two dimensions as (..., seq_len_q,
         dv) containing the scaled dot product attention
        @weights a tensor with its last two dimensions as (..., seq_len_q,
         seq_len_v) containing the attention weights

    """
    seq_len_q, dk = Q.shape[-2:]
    seq_len_v = K.shape[-2]
    dv = V.shape[-1]

    attn = tf.linalg.matmul(Q, K, transpose_b=True)
    attn = attn/tf.cast(dk, tf.float32)

    if mask is not None:
        mask *= -1e9
        attn += mask

    weights = tf.keras.backend.softmax(attn)
    output = tf.linalg.matmul(weights, V)

    return output, weights
