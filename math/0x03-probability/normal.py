#!/usr/bin/env python3
"""This modual contains the class created for Normal distribution"""


class Normal:
    """
        This class Represents the Normal distribution.

        Estimations used:
        e = 2.7182818285
        π = 3.1415926536
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        if not data:
            if stddev < 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            self.mean = sum(data)/len(data)
            variance = 0
            for i in data:
                variance += (i - self.mean)**2
            variance = variance/len(data)
            self.stddev = variance**.5
