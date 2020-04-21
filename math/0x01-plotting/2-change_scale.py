#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

plt.plot(x, y)  # plot relationship
plt.xlim(right=28650, left=0)  # set the x range
plt.title("Exponential Decay of C-14")  # title
plt.xlabel("Time (years)")  # xlabel
plt.ylabel("Fraction Remaining")  # ylabel
plt.yscale("log")  # scale of the yaxis
plt.show()  # render
