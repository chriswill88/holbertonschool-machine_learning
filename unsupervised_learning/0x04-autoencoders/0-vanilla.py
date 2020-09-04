#!/usr/bin/env python3
"""this module contians the function for task 0"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    this function creates an autoencoder
    with the various diffrent parts of the autoencoder

    @input_dims: the input dimensions for the autoencoder
    @hidden_layers: the hidden layers
    @latent_dims: the layer in the center
    """
    # auto
    input_img = keras.layers.Input((input_dims,))
    hiddenLen = len(hidden_layers)
    for i in range(len(hidden_layers)):
        out = input_img if i == 0 else enc
        enc = keras.layers.Dense(
            hidden_layers[i - hiddenLen], activation='relu')(out)

    latent = keras.layers.Dense(latent_dims, activation='relu')(enc)

    i = len(hidden_layers) - 1
    while i >= 0:
        out = latent if i == len(hidden_layers) - 1 else dec
        dec = keras.layers.Dense(hidden_layers[i], activation='relu')(out)
        i -= 1
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(dec)
    autoencode = keras.models.Model(input_img, decoded)

    # encoder
    encoder = keras.models.Model(input_img, latent)

    # decoder
    encoded_input = keras.layers.Input(shape=(latent_dims,))
    out = encoded_input
    for i in range(len(autoencode.layers)):
        if i > hiddenLen + 1:
            out = autoencode.layers[i](out)

    decoder = keras.models.Model(encoded_input, out)

    # comilation
    autoencode.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, autoencode
