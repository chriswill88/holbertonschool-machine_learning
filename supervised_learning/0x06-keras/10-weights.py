#!/usr/bin/env python3
"""this modual contains two function save/load weights"""
import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """save_weights - save weights"""
    network.save_weights(filename, save_format)


def load_weights(network, filename):
    """load_weights - load weights"""
    return network.load_weights(filename)
