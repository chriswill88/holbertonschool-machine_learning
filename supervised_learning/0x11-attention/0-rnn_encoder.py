#!/usr/bin/env python3
"""this module is for task 0"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """RNNEncoder - class for rnn"""
    def __init__(self, vocab, embedding, units, batch):
        super(RNNEncoder, self).__init__()

        self.batch = batch
        self.units = units

        self.embedding = tf.keras.layers.Embedding(
            vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units, return_sequences=True, return_state=True,
            recurrent_initializer='glorot_uniform')

    def call(self, x, initial):
        """call - pieces the layers together"""
        x = self.embedding(x)
        out, state = self.gru(x, initial_state=initial)
        return out, state

    def initialize_hidden_state(self):
        """Initializes the hidden states for the RNNto a tensor of zeros"""
        return tf.zeros((self.batch, self.units))
