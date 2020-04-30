#!/usr/bin/env python3
"""This modual contains the class created for task 0"""


class Poisson:
    """
    This class Represents the poisson distribution.
    """
    def __init__(self, data=None, lambtha=1.):
        self.lambtha = float(lambtha)

        if data is None:
            if lambtha < 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.data = lambtha
        else:
            # calculate the lambtha of data
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.lambtha = sum(data)/len(data)
