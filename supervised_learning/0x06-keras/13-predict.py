#!/usr/bin/env python3
"""this modual holds in the prediction function"""
import tensorflow.keras as K


def predict(network, data, verbose=False):
    """predict - returns the prediction"""
    return network.predict(data, verbose=verbose)
