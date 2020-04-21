#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

y = np.arange(0, 11) ** 3
plt.plot(y, 'r')
plt.xlim(right=10, left=0)  # adjust the left leaving right unchanged
plt.show()
