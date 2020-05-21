#!/usr/bin/env python3
"""this is the modual that contains the function for task 3"""
import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """
    y is a placeholder for the labels of the input data
    y_pred is a tensor containing networks predictions
    returns: tensor containing decimal accuracy of the prediction
    """
    loss = tf.losses.mean_squared_error(labels=y, predictions=y_pred)
    return loss
