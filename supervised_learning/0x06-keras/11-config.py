#!/usr/bin/env python3
"""this modual contains two function save/load models"""
import tensorflow.keras as K


def save_config(network, filename):
    """save_model - saves a model json"""
    netjson = network.to_json()
    with open(filename, "w") as files:
        files.write(netjson)


def load_config(filename):
    """load_model - load a model json"""
    with open(filename, 'r') as files:
        modeljson = files.read()
        return K.models.model_from_json(modeljson)
