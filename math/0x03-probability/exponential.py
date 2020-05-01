#!/usr/bin/env python3
"""This modual contains the class created for Exponential distribution"""


class Exponential:
    """
        This class Represents the Exponential distribution.

        Estimations used:
        e = 2.7182818285
        π = 3.1415926536
    """
    def __init__(self, data=None, lambtha=1.):
        if lambtha < 0:
            raise ValueError("lambtha must be a positive value")
        self.lambtha = float(lambtha)
        if data is None:
            self.data = lambtha
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data)/len(data)
            self.data = data
