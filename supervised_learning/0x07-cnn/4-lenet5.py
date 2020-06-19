#!/usr/bin/env python3
import tensorflow as tf
"""This function is used for task 4"""


def calculate_accuracy(y, y_pred):
    """
        y is a placeholder for the labels of the input data
        y_pred is a tensor containing networks predictions
        returns: tensor containing decimal accuracy of the prediction
    """
    mean = tf.math.reduce_mean

    max_y = tf.argmax(y, axis=1)
    max_yp = tf.argmax(y_pred, axis=1)

    eq = tf.equal(max_y, max_yp)

    avg = mean(tf.cast(eq, tf.float32))

    return avg


def calculate_loss(y, y_pred):
    """calculates the loss"""
    sm = tf.losses.softmax_cross_entropy
    return sm(y, y_pred)


def lenet5(x, y):
    """
        builds a modified version of the LeNet-5 architecture using tensorflow:

        x is a tf.placeholder of shape (m, 28, 28, 1) containing the input
         images for the network
            m is the number of images

        y is a tf.placeholder of shape (m, 10) containing the one-hot labels
         for the network

        The model should consist of the following layers in order:
            1.Convolutional layer with 6 kernels of shape 5x5 with same padding
            2.Max pooling layer with kernels of shape 2x2 with 2x2 strides
            3.Convolutional layer with 16 kernels of shape 5x5 with valid
             padding
            4.Max pooling layer with kernels of shape 2x2 with 2x2 strides
            5.Fully connected layer with 120 nodes
            6.Fully connected layer with 84 nodes
            7.Fully connected softmax output layer with 10 nodes

        All layers requiring initialization should initialize their kernels
         with the he_normal initialization method:
        tf.contrib.layers.variance_scaling_initializer()
        All hidden layers requiring activation should use the relu activation
        function

        you may import tensorflow as tf
        you may NOT use tf.keras
        Returns:
            a tensor for the softmax activated output
            a training operation that utilizes Adam optimization
             (with default hyperparameters)
            a tensor for the loss of the network
            a tensor for the accuracy of the network
    """
    # layer 1
    init = tf.contrib.layers.variance_scaling_initializer()
    conv_lay1 = tf.layers.Conv2D(
        filters=6, kernel_size=(5, 5), padding='same',
        kernel_initializer=init, activation='relu')
    first_out = conv_lay1(x)
    maxPool1 = tf.layers.MaxPooling2D(
        (2, 2), (2, 2))
    first_pool = maxPool1(first_out)

    # Layer 2
    conv_lay2 = tf.layers.Conv2D(
        filters=16, kernel_size=(5, 5),
        kernel_initializer=init, activation='relu')
    second_out = conv_lay2(first_pool)
    maxPool2 = tf.layers.MaxPooling2D(
        (2, 2), (2, 2))
    second_pool = maxPool2(second_out)

    # flatten
    flatten = tf.layers.Flatten()(second_pool)

    # Layer 3
    FC1 = tf.layers.Dense(120, kernel_initializer=init, activation='relu')
    FC_out3 = FC1(flatten)

    # Layer 4
    FC2 = tf.layers.Dense(84, kernel_initializer=init, activation='relu')
    FC_out4 = FC2(FC_out3)

    # Final Layer: tensor for the softmax layer
    FCF = tf.layers.Dense(10, kernel_initializer=init)
    final = FCF(FC_out4)

    # calc loss before doing softmax activation
    loss = calculate_loss(y, final)

    # seperate from last layer
    sm_layer = tf.nn.softmax(final)

    # accuracy
    acc = calculate_accuracy(y, sm_layer)

    # training operation that utilizes Adam optimization
    op = tf.train.AdamOptimizer()
    train = op.minimize(loss)

    return sm_layer, train, loss, acc
