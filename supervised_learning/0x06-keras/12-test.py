#!/usr/bin/env python3


def test_model(network, data, labels, verbose=True):
    """test_models: """
    results = network.evaluate(data. labels, verbose=verbose)
    return results
