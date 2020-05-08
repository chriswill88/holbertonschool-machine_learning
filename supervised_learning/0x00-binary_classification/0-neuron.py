#!/usr/bin/env python3
import numpy as np
"""
    this modual holds the class created for task 0
"""


class Neuron:
    """
        Neuron - class for a neuron
        nx = the number of input freatures to the neuron
    """
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.standard_normal((1, 784))
        self.b = 0
        self.A = 0
