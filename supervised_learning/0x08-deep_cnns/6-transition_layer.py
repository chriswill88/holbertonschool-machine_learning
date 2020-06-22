#!/usr/bin/env python3
"""This modual contains the function for task 6"""
import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
        Transition layer as described in
         Densely Connected Convolutional Networks:

        @X is the output from the previous layer
        @nb_filters is an integer representing the number of filters in X
        @compression is the compression factor for the transition layer

        Your code should implement compression as used in DenseNet-C
        All weights should use he normal initialization
        All convolutions should be preceded by Batch Normalization
         and a rectified linear activation (ReLU), respectively
        Returns: The output of the transition layer and the number
         of filters within the output, respectively
    """
    init = K.initializers.he_normal()

    op = K.layers.BatchNormalization(3)(X)
    op = K.layers.Activation('relu')(op)
    op = K.layers.Conv2D(
        int(nb_filters * compression), (1, 1),
        padding='same', kernel_initializer=init)(op)
    op = K.layers.AveragePooling2D((2, 2))(op)

    return op, int(nb_filters * compression)
