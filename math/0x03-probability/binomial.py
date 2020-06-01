#!/usr/bin/env python3
"""this modual contains a class binomial"""


class Binomial:
    """Binomial is the class for binomial distribution"""
    def __init__(self, data=None, n=1, p=0.5):
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if 0 >= p and p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            win = []
            i, wins, loss = 0, 0, 0
            while i < len(data):
                if data[i] > data[i + 1]:
                    loss += 1
                else:
                    win.append(data[i])
                    wins += 1
                i += 2

            mean = sum(data)/len(data)
            self.n = int(len(data)/2)
            self.p = float(mean/self.n)
