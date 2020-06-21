#!/usr/bin/env python3
"""this modual contains the function used in task 2"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
        builds an identity block as described in
         Deep Residual Learning for Image Recognition (2015):

        A_prev is the output from the previous layer
        filters is a tuple or list containing F11, F3, F12, respectively:
        F11 is the number of filters in the first 1x1 convolution
        F3 is the number of filters in the 3x3 convolution
        F12 is the number of filters in the second 1x1 convolution
        All convolutions inside the block should be followed by batchs
         normalization
         along the channels axis and a rectified linear activation (ReLU),
          respectively.
        All weights should use he normal initialization
        Returns: the activated output of the identity block
    """
    init = K.initializers.he_normal()

    ep = K.layers.Conv2D(
        filters[0], (1, 1), (1, 1),
        kernel_initializer=init)(A_prev)
    ep = K.layers.BatchNormalization(3)(ep)
    ep = K.layers.Activation('relu')(ep)

    ep = K.layers.Conv2D(
        filters[1], (3, 3), (1, 1), padding='same',
        kernel_initializer=init)(ep)
    ep = K.layers.BatchNormalization(3)(ep)
    ep = K.layers.Activation('relu')(ep)

    ep = K.layers.Conv2D(
        filters[2], (1, 1),
        kernel_initializer=init)(ep)
    ep = K.layers.BatchNormalization(3)(ep)

    ep = K.layers.Add()([ep, A_prev])
    ep = K.layers.Activation('relu')(ep)

    return ep
