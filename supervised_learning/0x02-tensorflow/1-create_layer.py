#!/usr/bin/env python3
"""modual for task 1"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    prev: the tensor output of the previous layer
    n: is the number of nodes in a layer to create
    activation: is the activation func. the layer should use

    layer should be given the name layer
    return: the tensor output layer
    """
    act = activation
    sess = tf.Session()
    weight = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    linear_model = tf.layers.Dense(
        units=n, activation=act, name='layer', kernel_initializer=weight)
    y = linear_model(prev)
    return y
