#!/usr/bin/env python3
import tensorflow as tf
from tensorflow import keras


def build_model(nx, layers, activations, lambtha, keep_prob):
    model = keras.Sequential()
    
    for i in range(len(layers)):
        model.add(layers.Dense(layers[i], activation=activations[i]))

    return model
