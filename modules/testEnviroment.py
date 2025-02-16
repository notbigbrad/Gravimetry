# DO NOT ALTER
# FOR TESTING PURPOSES

import numpy as np
import scipy
import os
import matplotlib.pyplot as plt
import time

import scipy.integrate

def f(x):
    return (x**2)

t = np.linspace(0,10,500)

plt.figure(figsize=[10,5])
plt.title("Test plot")
plt.plot(t, f(t))
plt.show()