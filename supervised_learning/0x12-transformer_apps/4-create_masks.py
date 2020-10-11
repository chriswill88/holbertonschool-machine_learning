#!/usr/bin/env python3
"""this module is created for task 4"""
import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """
    That creates all masks for training/validation:

    @inputs: is a tf.Tensor of shape (batch_size, seq_len_in) that contains the input sentence
    @target: is a tf.Tensor of shape (batch_size, seq_len_out) that contains the target sentence

    This function should only use tensorflow operations in order to properly function in the training step
    Returns: encoder_mask, look_ahead_mask, decoder_mask
        encoder_mask is the tf.Tensor padding mask of shape (batch_size, 1, 1, seq_len_in) to be applied in the encoder
        look_ahead_mask is the tf.Tensor look ahead mask of shape (batch_size, 1, seq_len_out, seq_len_out) to be applied in the decoder
        decoder_mask is the tf.Tensor padding mask of shape (batch_size, 1, 1, seq_len_in) to be applied in the decoder
    """
    batch_size, seq_len_in = inputs.get_shape()
    _, seq_len_out = target.get_shape()

    encoder_mask = tf.cast(tf.math.equal(
        inputs, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones(
        (seq_len_out, seq_len_out)), -1, 0)
    look_ahead_mask = tf.maximum(tf.cast(tf.math.equal(
        target, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :],
        look_ahead_mask)
    decoder_mask = tf.cast(tf.math.equal(
        inputs, 0), tf.float32)[:, tf.newaxis, tf.newaxis, :]
    return encoder_mask, look_ahead_mask, decoder_mask
