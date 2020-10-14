#!/usr/bin/env python3
"""this module contains the function for task 0"""
import numpy as np
import tensorflow as tf


def generator(Z):
    """this function creates a generator and returns a tensor"""
    with tf.variable_scope("generator", reuse=tf.AUTO_REUSE):
        layer_1 = tf.keras.layers.Dense(128, 'relu',)(Z)
        layer_2 = tf.keras.layers.Dense(784, 'sigmoid')(layer_1)
        layer_1.name("Layer_1")
        layer_2.name("Layer_2")
        model = tf.keras.Model(inputs=Z, outputs=layer_2)
        x = model(Z)
    return x
