#!/usr/bin/env python3
"""task 2 function"""
import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """
    network is the model to optimize
    alpha is the learning rate
    beta1 is the first Adam optimization parameter
    beta2 is the second Adam optimization parameter
    """
    network.compile(
        optimizer=K.optimizers.Adam(alpha, beta1, beta2),
        loss='categorical_crossentropy',
        metrics=["accuracy"]
    )
