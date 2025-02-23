# DO NOT EDIT
# ONLY MAKE CHANGES TO COPY

import numpy as np
import matplotlib.pyplot as plt

def physicalPendulum(t, A0, gamma, o, phi):
    exponent = -(t*gamma)
    a = (o*t - phi)
    return (A0*np.exp(exponent)*np.cos(a))

def sin(t,o,p):
    return np.sin(p + o*t)

def demo():
    amplitude = 1
    dampingFactor = 1
    gravity = 9.81616
    length = 0.25
    o = np.sqrt(gravity/length)
    angle = 0
    x = np.linspace(0,25,10000)
    plt.figure(figsize=[5,10])
    plt.plot(x,physicalPendulum(x, amplitude, dampingFactor, o, angle))
    plt.show()