#!/usr/bin/env python3
"""this modual contains the function for task 6"""
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """
    Creates the training operation for a neural network in
    tensorflow using the gradient descent with momentum
    optimization algorithm:

    @loss:
        is the loss of the network
    @alpha:
        is the learning rate
    @beta1:
        is the momentum weight
    @Returns:
        the momentum optimization operation
    """
    moment = tf.train.MomentumOptimizer(alpha, beta1)
    return moment.minimize(loss)
