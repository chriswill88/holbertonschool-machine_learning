#!/usr/bin/env python3
"""This modual contains the class created for task 0"""


class Poisson:
    """
        This class Represents the poisson distribution.

        Estimations used:
        e = 2.7182818285
        π = 3.1415926536
    """
    def __init__(self, data=None, lambtha=1.):
        if lambtha < 1:
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

    def pmf(self, k):
        """
            Calculates the Probability Mass Function
            for the successes (k) given.
        """
        k, fact, l, d = int(k), 1, self.lambtha, self.data
        if k < 0:
            return 0
        for i in range(1, k + 1):
            fact *= i
        return (l**k * 2.7182818285**(l * -1))/(fact)

    def cdf(self, k):
        """
            Calculates the value of the CDF
            for a given number of “successes”.
        """
        k = int(k)
        if k < 0:
            return 0
        return k/1
