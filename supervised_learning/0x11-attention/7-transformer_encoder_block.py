#!/usr/bin/env python3
"""this module contains a function for task 5"""
import tensorflow as tf
MultiHeadAttention = __import__('6-multihead_attention').MultiHeadAttention


class EncoderBlock(tf.keras.layers.Layer):
    """This class represents an encoder block for a transformer"""
    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """EncoderBlock represents an encoder block for a transformer"""
        super(EncoderBlock, self).__init__()

        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(hidden, 'relu')
        self.dense_output = tf.keras.layers.Dense(dm)

        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)

        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """creates the Encoder block model"""
        attn_output, _ = self.mha(x, x, x, mask)
        # (batch_size, input_seq_len, dm)
        attn_output = self.ffn(attn_output)

        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)
        # (batch_size, input_seq_len, dm)

        ffn_output = self.dense_output(self.dense_hidden(out1))  # (batch_size, target_seq_len, dm)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        # (batch_size, input_seq_len, dm)
        return out2
