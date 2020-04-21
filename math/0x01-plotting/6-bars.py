#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5)
fruit = np.random.randint(0, 20, (4, 3))

x = np.arange(3)

width = .5

c = ["Farrah", "Fred", "Felicia"]
rows = ["apples", "bananas", "oranges", "peaches"]

a = fruit[0]
b = fruit[1]
o = fruit[2]
p = fruit[3]

plt.bar(c, a, .5, align="center", color='r', label="apples")
plt.bar(c, b, .5, a, align="center", color='yellow', label="bananas")
plt.bar(c, o, .5, a+b, align="center", color="#ff8000", label="oranges")
plt.bar(c, p, .5, a+b+o, align="center", color="#ffe5b4", label="peaches")
plt.legend()

plt.ylabel("Quantity of Fruit")
plt.title("Number of Fruit per Person")
plt.ylim(0, 80, 10)

plt.show()
