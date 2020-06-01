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
        if data is None:
            if stddev < 1:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = float(sum(data)/len(data))
            variance = 0
            for i in data:
                variance += (i - self.mean)**2
            variance = variance/len(data)
            self.stddev = float(variance**.5)

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean)/self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return (self.stddev * z) + self.mean

    def pdf(self, x):
        """Calculates the value of the PDF for a given x-value"""
        s = self.stddev
        m = self.mean
        pi = 3.1415926536
        e = 2.7182818285

        first = 1/(s*(2*pi)**.5)
        second = e**(-.5*((x - m) / s)**2)
        return first * second
        # return (1/(m * (2 * pi)**.5)) * e**(-.5*(((x - m) / s)**2))

    def cdf(self, x):
        """Calculates the value of the CDF for a given x-value"""
        mean = self.mean
        std = self.stddev
        variance = std**2

        pi = 3.1415926536
        xv = (x - mean)/(std * 2**(.5))
        erf = 2/(pi**.5)*(xv - (xv**3)/3+(xv**5)/10-(xv**7)/42+(xv**9)/216)
        c = (1/2) * (1 + erf)

        return c
