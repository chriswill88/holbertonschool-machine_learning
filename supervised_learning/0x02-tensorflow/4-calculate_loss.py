#!/usr/bin/env python3
"""this modual contains the function for taks 4"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """calculates the loss"""
    sm = tf.losses.softmax_cross_entropy
    return sm(y, y_pred)
