#!/usr/bin/env python3
"""this modual contains the functon for task 1"""
import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """
        build the inception network as described in
        Going Deeper with Convolutions(2014):

        You can assume the input data will have shape (224, 224, 3)
        All convolutions inside and outside the inception block should
         use a rectified linear activation (ReLU)
        Use inception_block from task 0
        Returns: the keras model
    """
    take = [64, 96, 128, 16, 32, 32]

    inputs = K.Input((224, 224, 3))

    # Convolution
    ep = K.layers.Conv2D(
        64, kernel_size=(7, 7), strides=(2, 2), activation='relu'
        )(inputs)
    # Max Pool
    ep = K.layers.MaxPool2D(
        (3, 3), (2, 2), "same")(ep)

    # C
    ep = K.layers.Conv2D(
        64, kernel_size=(1, 1), activation='relu'
        )(ep)
    ep = K.layers.Conv2D(
        192, kernel_size=(3, 3), padding='same', activation='relu'
        )(ep)

    # MP
    ep = K.layers.MaxPool2D((3, 3), (2, 2), "same")(ep)

    # Inception 2X
    for i in range(2):
        ep = inception_block(ep, take)

    # MP
    ep = K.layers.MaxPool2D(
        (3, 3), (2, 2), "same")(ep)

    # Inc 5X
    for i in range(5):
        ep = inception_block(ep, take)

    # Mp
    ep = K.layers.MaxPool2D(
        (3, 3), (2, 2), "same")(ep)

    # Inc 2X
    for i in range(2):
        ep = inception_block(ep, take)

    # AP
    ep = K.layers.AveragePooling2D(
        (7, 7), (1, 1))(ep)

    # Dropout
    ep = K.layers.Dropout(.4)(ep)

    # linear
    ep = K.layers.Dense(1000, activation='softmax')(ep)

    model = K.Model(inputs, ep)
    return model
