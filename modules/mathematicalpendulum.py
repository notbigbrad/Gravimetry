# DO NOT EDIT
# ONLY MAKE CHANGES TO COPY
# LEAVE THIS CODE FOR TESTING PURPOSES FOR LATER USE

import numpy as np
import scipy
import matplotlib.pyplot as plt

def mathPendulum(t, A0, gamma, g, l, phi):
    exponent = -(t/(4*gamma))
    a = ((g/l)*t - phi)
    return (A0*np.exp(exponent)*np.cos(a))

def demo():
    amplitude = 1
    dampingFactor = 1
    gravity = 9.81616
    length = 0.25
    angle = 0
    x = np.linspace(0,25,10000)
    plt.figure(figsize=[5,10])
    plt.plot(x,mathPendulum(x, amplitude, dampingFactor, gravity, length, angle))
    plt.show()