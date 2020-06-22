#!/usr/bin/env python3
"""This modual contains the function used for task 7"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
        The DenseNet-121 architecture as described
         in Densely Connected Convolutional Networks:

        @growth_rate is the growth rate
        @compression is the compression factor

        You can assume the input data will have shape (224, 224, 3)
        All convolutions should be preceded by Batch Normalization
         and a rectified linear activation (ReLU), respectively
        All weights should use he normal initialization

        Return the keras model
    """
    ip = 64
    layers = [6, 12, 24, 16]

    inp = K.Input((224, 224, 3))
    init = K.initializers.he_normal()

    op = K.layers.BatchNormalization(3)(inp)
    op = K.layers.Activation('relu')(op)
    op = K.layers.Conv2D(
        64, (7, 7), (2, 2),
        padding='same', kernel_initializer=init)(op)

    op = K.layers.MaxPool2D(
        (3, 3), (2, 2), padding='same')(op)

    for i in range(4):
        op, ip = dense_block(
            op, ip, growth_rate, layers[i])
        if i != 3:
            op, ip = transition_layer(op, ip, compression)

    op = K.layers.AveragePooling2D((7, 7))(op)
    op = K.layers.Dense(1000, activation='softmax')(op)

    model = K.Model(inp, op)
    return model
