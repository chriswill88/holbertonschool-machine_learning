    #!/usr/bin/env python3
"""This modual holds the class created for task 0"""
import numpy as np


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
        self.__W = np.random.randn(1, nx)
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        return self.__W

    @property
    def b(self):
        return self.__b

    @property
    def A(self):
        return self.__A
