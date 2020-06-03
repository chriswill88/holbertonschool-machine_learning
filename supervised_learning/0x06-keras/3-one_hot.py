#!/usr/bin/env python3
"""simple one hot matrix with keras"""
import tensorflow.keras as K


def one_hot(labels, classes=None):
    """one_hot matrix made with keras"""
    return K.utils.to_categorical(labels, classes)
