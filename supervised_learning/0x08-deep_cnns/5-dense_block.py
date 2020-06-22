#!/usr/bin/env python3
"""This modual contains the function used in task 5"""
import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
        dense block as described in Densely Connected Convolutional Networks:

        @X is the output from the previous layer
        @nb_filters is an integer representing the number of filters in X
        @growth_rate is the growth rate for the dense block
        @layers is the number of layers in the dense block

        You should use the bottleneck layers used for DenseNet-B
        All weights should use he normal initialization
        All convolutions should be preceded by Batch Normalization
         and a rectified linear activation (ReLU), respectively
        Returns: The concatenated output of each layer within
         the Dense Block and the number of filters within the
         concatenated outputs, respectively
    """
    init = K.initializers.he_normal()
    nb = nb_filters
    inp = X

    for i in range(layers):
        op = K.layers.BatchNormalization(3)(inp)
        op = K.layers.Activation('relu')(op)
        op = K.layers.Conv2D(
            growth_rate * 4, (1, 1), padding='same',
            kernel_initializer=init)(op)

        op = K.layers.BatchNormalization(3)(op)
        op = K.layers.Activation('relu')(op)
        op = K.layers.Conv2D(
            growth_rate, (3, 3), padding='same',
            kernel_initializer=init)(op)

        op = K.layers.Concatenate()([X, op])

        X = op
        inp = op
        nb += growth_rate

    return op, nb
