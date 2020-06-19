#!/usr/bin/env python3
"""This function is used for task 5"""
import tensorflow.keras as K


def lenet5(X):
    """
        that builds a modified version of the LeNet-5 architecture using keras:

        X is a K.Input of shape (m, 28, 28, 1) containing the input images for
         the network
            m is the number of images
        The model should consist of the following layers in order:
            Convolutional layer with 6 kernels of shape 5x5 with same padding
            Max pooling layer with kernels of shape 2x2 with 2x2 strides
            Convolutional layer with 16 kernels of shape 5x5 with valid padding
            Max pooling layer with kernels of shape 2x2 with 2x2 strides
            Fully connected layer with 120 nodes
            Fully connected layer with 84 nodes
            Fully connected softmax output layer with 10 nodes

        All layers requiring initialization should initialize their kernels
         with the he_normal initialization method

        All hidden layers requiring activation should use the relu activation
         function

        you may import tensorflow.keras as K
        Returns: a K.Model compiled to use Adam optimization
         (with default hyperparameters) and accuracy metrics
    """

    ini = K.initializers.he_normal()

    l1_conv = K.layers.Conv2d(
        6, (5, 5), padding='same',
        activation='relu', kernel_initializer=ini)(x)

    l1_mp = k.layers.MaxPool2D(
        (2, 2), (2, 2))(l1_conv)

    l2_conv = K.layers.Conv2d(
        16, (5, 5), padding='valid',
        activation='relu', kernel_initializer=ini)(l1_mp)

    l2_mp = K.layers.MaxPool2D(
        (2, 2), (2, 2))(l2_conv)

    compress = K.layers.Flatten()(l2_mp)

    fc1 = K.layers.Dense(
        120, activation='relu', kernel_initializer=ini)(compress)
    fc2 = K.layers.Dense(
        84, activation='relu', kernel_initializer=ini)(fc1)
    f = K.layers.Dense(
        10, activation='softmax', kernel_initializer=ini)(fc2)

    model = K.Model(inputs=X, output=f)

    return model.compile(
        optimizer=K.optimizers.Adam(),
        metrics=["accuracy"]
    )
