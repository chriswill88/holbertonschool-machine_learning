#!/usr/bin/env python3
"""this module contians the function for task 1"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """
    this function creates an sparse autoencoder
    with the various diffrent parts of the autoencoder

    @input_dims: the input dimensions for the autoencoder
    @hidden_layers: the hidden layers
    @latent_dims: the layer in the center
    """
    # encoded
    input_img = keras.layers.Input((input_dims,))
    L1 = keras.regularizers.l1(lambtha)
    hiddenLen = len(hidden_layers)
    out = input_img
    for i in range(len(hidden_layers)):
        out = keras.layers.Dense(
            hidden_layers[i - hiddenLen], activation='relu')(out)

    latent = keras.layers.Dense(latent_dims, activation='relu', activity_regularizer=L1)(out)
    encode = keras.models.Model(input_img, latent)

    # decoded
    dec_input = keras.layers.Input((latent_dims,))
    out = dec_input
    i = len(hidden_layers) - 1
    while i >= 0:
        out = keras.layers.Dense(hidden_layers[i], activation='relu')(out)
        i -= 1
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(out)
    decode = keras.models.Model(dec_input, decoded)

    # auto
    encoder = encode(input_img)
    decoder = decode(encoder)
    autoencode = keras.models.Model(input_img, decoder)

    autoencode.compile(optimizer='adam', loss='binary_crossentropy')
    return encode, decode, autoencode
