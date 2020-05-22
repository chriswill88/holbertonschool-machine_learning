#!/usr/bin/env python3
"""this modual contains the function for task 5"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """create_train_op - creates a training operation"""
    opt = tf.train.GradientDescentOptimizer(alpha)
    train = opt.minimize(loss)
    return train
