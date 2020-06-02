#!/usr/bin/env python3
import warnings
with warnings.catch_warnings():  
    warnings.filterwarnings("ignore",category=FutureWarning)
    import tensorflow as tf
    from tensorflow import keras

build_model = __import__('0-sequential').build_model

if __name__ == '__main__':
    network = build_model(784, [256, 256, 10], ['tanh', 'tanh', 'softmax'], 0.001, 0.95)
    network.summary()
    print(network.losses)