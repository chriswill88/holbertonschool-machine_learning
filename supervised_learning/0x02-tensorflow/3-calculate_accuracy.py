#!/usr/bin/env python3
"""this is the modual that contains the function for task 3"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    y is a placeholder for the labels of the input data
    y_pred is a tensor containing networks predictions
    returns: tensor containing decimal accuracy of the prediction
    """
    mean = tf.math.reduce_mean

    max_y = tf.argmax(y, axis=1)
    max_yp = tf.argmax(y_pred, axis=1)

    eq = tf.equal(max_y, max_yp)

    avg = mean(tf.cast(eq, tf.float32))

    return avg
