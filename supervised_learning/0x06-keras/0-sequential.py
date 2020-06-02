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
    model = K.Sequential()

    # creating regulizer variable - applies to every layer
    L2 = K.regularizers.l2(lambtha)

    # adding first layer with input shape
    ip = K.layers.Dense(
        layers[0],
        activation=activations[0],
        input_shape=(nx,),
        kernel_regularizer=L2
        )
    model.add(ip)

    i = 1
    while i < len(layers):
        # dropout goes after first layer
        # and before other layers are added
        model.add(K.layers.Dropout(1 - keep_prob))

        # adding depth to neural network - adding layers
        model.add(K.layers.Dense(
            layers[i],
            activation=activations[i],
            kernel_regularizer=L2
        ))
        i += 1

    return model
