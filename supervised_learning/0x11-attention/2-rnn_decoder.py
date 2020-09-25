#!/usr/bin/env python3
"""this module is for task 2"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """RNNDecoder - class for rnn"""
    def __init__(self, vocab, embedding, units, batch):
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(
            vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units, return_sequences=True, return_state=True,
            recurrent_initializer='glorot_uniform')
        self.F = tf.layers.Dense(vocab)
        self.att = SelfAttention(units)

    def call(self, x, s_prev, hidden_states):
        """this function pieces together the decoder"""
        con_v, att_w = self.att(s_prev, hidden_states)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(con_v, 1), x], axis=-1)
        output, state = self.gru(x)
        output = tf.reshape(output, (-1, output.shape[2]))
        x = self.F(output)
        return x, state
