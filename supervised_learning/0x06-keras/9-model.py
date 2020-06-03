#!/usr/bin/env python3
"""this modual contains two function save/load models"""
import tensorflow.keras as K


def save_model(network, filename):
    """save_model - saves a model"""
    network.save(filename)


def load_model(filename):
    """load_model - load a model """
    return K.models.load_model(filename)
