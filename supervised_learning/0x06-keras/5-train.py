#!/usr/bin/env python3
"""
this function trains a model using minibatch gradient decent
"""
import tensorflow.keras as K


def train_model(
    network, data, labels,
    batch_size, epochs,
    validation_data=None,
    verbose=True, shuffle=False
        ):
    """
    network is the model to train
    data is a numpy.ndarray of shape (m, nx) containing the input data
    labels one-hot numpy.ndarray shape (m, classes) containing labels
    batch_size is the size of the batch used for mini-batch gradient descent
    epochs is the number of passes through data for mini-batch gradient descent
    verbose boolean that determines if output should be printed during training
    shuffle boolean that determines whether to shuffle batches every epoch.
    Normally, it is a good idea to shuffle, but for reproducibility,
    we have chosen to set the default to False.
    validation_data is the data to validate the model with, if not None
    Returns: the History object generated after training the model
    """
    return network.fit(
        data, labels,
        batch_size=batch_size, epochs=epochs,
        verbose=verbose, shuffle=shuffle,
        validation_data=validation_data)
