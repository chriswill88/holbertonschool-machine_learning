#!/usr/bin/env python3
"""This modual contiains a function that uses keras"""
import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    nx is the number of input features to the network
    layers - list containing the number of nodes in each layer of the network
    activations - the activation functions used for each layer of the network
    lambtha is the L2 regularization parameter
    keep_prob is the probability that a node will be kept for dropout
    """

    # initializing the model - first step
    inputs = K.Input(shape=(nx,))

    # creating regulizer variable - applies to every layer
    L2 = K.regularizers.l2(lambtha)

    # adding first layer with input shape
    X = K.layers.Dense(
        layers[0],
        activation=activations[0],
        kernel_regularizer=L2
    )(inputs)
    i = 1
    while i < len(layers):
        # dropout goes after first layer
        # and before other layers are added
        X = K.layers.Dropout(1 - keep_prob)(X)

        # adding depth to neural network - adding layers
        X = K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=L2
        )(X)
        i += 1
    model = K.Model(inputs=inputs, outputs=X)
    return model
