#!/usr/bin/env python3
"""this modual contains the functon for task 0"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    builds an inception block as described in Going Deeper with Convolutions (2014):

        @A_prev
            is the output from the previous layer
        @filters
            is a tuple or list containing F1, F3R, F3,F5R, F5, FPP
            respectively:
            * F1 is the number of filters in the 1x1 convolution
            * F3R is the number of filters in the 1x1 convolution before the
             3x3 convolution
            * F3 is the number of filters in the 3x3 convolution
            * F5R is the number of filters in the 1x1 convolution before the
             5x5 convolution
            * F5 is the number of filters in the 5x5 convolution
            * FPP is the number of filters in the 1x1 convolution after the
             max pooling
        All convolutions inside the inception block should use a
         linear activation (ReLU)
        Returns: the concatenated output of the inception block
    """
    F1, F3R, F3,F5R, F5, FPP = filters
    
    E1 = K.layers.Conv2D(
        F1, (1, 1), activation='relu',
        )(A_prev)

    L2 = K.layers.Conv2D(
        F3R, (1, 1), activation='relu',
        )(A_prev)
    E2 = K.layers.Conv2D(
        F3, (3, 3),  activation='relu', padding="same"
        )(L2)

    L3 = K.layers.Conv2D(
        F5R, (1, 1), activation='relu',
        )(A_prev)
    E3 = K.layers.Conv2D(
        F5, (5, 5),  activation='relu', padding="same"
        )(L3)

    L4 = K.layers.MaxPool2D((3, 3), (1, 1), padding='same'
        )(A_prev)
    E4 = K.layers.Conv2D(
        FPP, (1, 1), activation='relu',
        )(L4)

    merge = K.layers.Concatenate()([E1, E2, E3, E4])
    return merge
