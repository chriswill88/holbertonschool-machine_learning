#!/usr/bin/enV python3
"""this module contains a function for task 4"""
import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """
    sdp_attention -
        this function calculates the scaled dot product attention
        @Q: is a tensor with its last two dimensions as (..., seq_len_q, dk)
         containing the query matrix
        @K: is a tensor with its last two dimensions as (..., seq_len_V, dk)
         containing the key matrix
        @V: is a tensor with its last two dimensions as (..., seq_len_V, dV)
         containing the Value matrix
        @mask: is a tensor that can be broadcast into (..., seq_len_q,
         seq_len_V) containing the optional mask, or defaulted to None
    Returns: output, weights
        @output a tensor with its last two dimensions as (..., seq_len_q,
         dV) containing the scaled dot product attention
        @weights a tensor with its last two dimensions as (..., seq_len_q,
         seq_len_V) containing the attention weights

    """
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    # (..., seq_len_q, seq_len_k)
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    output = tf.matmul(attention_weights, V)
    return output, attention_weights
