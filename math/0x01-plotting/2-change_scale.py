#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, 28651, 5730)
r = np.log(0.5)
t = 5730
y = np.exp((r / t) * x)

plt.plot(x, y, range(0, 28650), scaley="log")  # plot relationship
plt.title("Exponential Decay of C-14")  # title
plt.xlabel("Time (years)")  # xlabel
plt.ylabel("Fraction Remaining")  # ylabel
plt.show()  # render
