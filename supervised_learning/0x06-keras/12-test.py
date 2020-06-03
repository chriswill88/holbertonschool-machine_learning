#!/usr/bin/env python3
"""this modual - contains function for task 12"""
import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """test_models: """
    results = network.evaluate(data. labels, verbose=verbose)
    return results
