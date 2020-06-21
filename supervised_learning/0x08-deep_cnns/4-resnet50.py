#!/usr/bin/env python3
"""This modual contians the function for task 4"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
        ResNet-50 architecture as described in Deep Residual Learning
        for Image Recognition (2015):

        You can assume the input data will have shape (224, 224, 3)
        All convolutions inside and outside the blocks should
         be followed by batch normalization along the channels axis
         and a rectified linear activation (ReLU), respectively.
        All weights should use he normal initialization

        Returns: the keras model
    """
    filters = [
        [64, 64, 256], [128, 128, 512], [256, 256, 1024], [512, 512, 2048]]

    inp = K.Input((224, 224, 3))

    op = K.layers.Conv2D(64, (7, 7), (2, 2))(inp)
    op = K.layers.BatchNormalization(3)(op)
    op = K.layers.Activation('relu')(op)

    op = K.layers.MaxPool2D((3, 3), (2, 2))(op)

    op = projection_block(op, filters[0])
    for i in range(2):
        op = identity_block(op, filters[0])

    op = projection_block(op, filters[1])
    for i in range(3):
        op = identity_block(op, filters[1])

    op = projection_block(op, filters[2])
    for i in range(5):
        op = identity_block(op, filters[2])

    op = projection_block(op, filters[3])
    for i in range(2):
        op = identity_block(op, filters[3])

    op = K.layers.AveragePooling2D((1, 1))(op)
    op = K.layers.Dense(1000, activation='softmax')(op)

    model = K.Model(inp, op)
    return model
