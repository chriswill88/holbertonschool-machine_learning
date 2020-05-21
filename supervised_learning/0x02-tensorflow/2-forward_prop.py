#!/usr/bin/env python3
"""this modual contains the function for task 2"""
create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    x - placeholder for input data
    layer_sizes - contain the number of nodes in each layer of the network
    activations - the activation functions for each layer of the network
    Return: is a list containing the prediction of network in tensor form
    """

    for i in range(len(layer_sizes)):
        x = create_layer(x, layer_sizes[i], activations[i])
    return x
