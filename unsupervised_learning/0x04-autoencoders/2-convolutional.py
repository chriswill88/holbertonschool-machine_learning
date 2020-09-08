#!/usr/bin/env python3
"""this module contians the function for task 1"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """
    this function creates an convolutional autoencoder
    with the various diffrent parts of the autoencoder

    @input_dims: the input dimensions for the autoencoder
    @filters: the hidden conv layers
    @latent_dims: the layer in the center
    """
    # encoded
    input_img = keras.layers.Input(shape=input_dims)

    hiddenLen = len(filters)
    print("hiddenLen", hiddenLen)
    out = input_img

    len_list = range(hiddenLen)
    rev_list = reversed(len_list)

    for i in len_list:
        print(i, filters[i])
        out = keras.layers.Conv2D(
            filters[i],
            (3, 3),
            padding='same',
            activation='relu'
        )(out)
        out = keras.layers.MaxPooling2D((2, 2), padding='same')(out)

    # decoded
    dec_input = keras.layers.Input(latent_dims)
    out = dec_input
    for i in rev_list:
        print(i, filters[i])
        if i == 0:
            out = keras.layers.Conv2D(
                filters[i],
                (3, 3),
                padding='valid',
                activation='relu'
            )(out)
        else:
            out = keras.layers.Conv2D(
                filters[i],
                (3, 3),
                padding='same',
                activation='relu'
            )(out)
        out = keras.layers.UpSampling2D((2, 2))(out)

    decoded = keras.layers.Conv2D(
        input_dims[-1], (3, 3), padding='same', activation='sigmoid')(out)

    decode = keras.models.Model(dec_input, decoded)

    print("decode")
    decode.summary()
    # auto
    encoder = encode(input_img)
    decoder = decode(encoder)
    autoencode = keras.models.Model(input_img, decoder)

    autoencode.compile(optimizer='adam', loss='binary_crossentropy')
    return encode, decode, autoencode
