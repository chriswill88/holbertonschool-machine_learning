#!/usr/bin/env python3
"""This modual contains the function used for task 8"""
import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """
        Creates the training operation for a neural network in tensorflow
        using the RMSProp optimization algorithm:

        @loss
            is the loss of the network
        @alpha
            is the learning rate
        @beta2
            is the RMSProp weight
        @epsilon
            is a small number to avoid division by zero

        Returns: the RMSProp optimization operation
    """
    rms = tf.train.RMSPropOptimizer(
        learning_rate=alpha, momentum=beta2, epsilon=epsilon)
    return rms.minimize(loss)
