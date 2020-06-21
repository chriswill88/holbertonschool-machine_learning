#!/usr/bin/env python3
"""This modual contains the function used in task 3"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
        uilds a projection block as described in Deep Residual Learning for
         Image Recognition (2015):

        @A_prev is the output from the previous layer
        @filters is a tuple or list containing F11, F3, F12, respectively:
            F11 is the number of filters in the first 1x1 convolution
            F3 is the number of filters in the 3x3 convolution
            F12 is the number of filters in the second 1x1 convolution as
             well as the 1x1 convolution in the shortcut connection
        @s is the stride of the first convolution in both the main path
         and the shortcut connection
        All convolutions inside the block should be followed by batch
         normalization along the channels axis and a rectified linear
         activation (ReLU), respectively.
        All weights should use he normal initialization
        Returns: the activated output of the projection block
    """
    init = K.initializers.he_normal()

    F11 = filters[0]
    F3 = filters[1]
    F12 = filters[2]

    ep = K.layers.Conv2D(
        F11, (1, 1), (s, s),
        kernel_initializer=init)(A_prev)
    ep = K.layers.BatchNormalization(3)(ep)
    ep = K.layers.Activation('relu')(ep)

    ep = K.layers.Conv2D(
        F3, (3, 3), (1, 1), padding='same',
        kernel_initializer=init)(ep)
    ep = K.layers.BatchNormalization(3)(ep)
    ep = K.layers.Activation('relu')(ep)

    ep = K.layers.Conv2D(
        F12, (1, 1),
        kernel_initializer=init)(ep)
    op = K.layers.Conv2D(
        F12, (1, 1), (s, s),
        kernel_initializer=init)(A_prev)

    ep = K.layers.BatchNormalization(3)(ep)
    op = K.layers.BatchNormalization(3)(op)

    ep = K.layers.Add()([ep, op])
    ep = K.layers.Activation('relu')(ep)

    return ep
