# DO NOT EDIT
# ONLY MAKE CHANGES TO COPY

import numpy as np
import scipy
import matplotlib.pyplot as plt

def physicalPendulum(t, A0, gamma, o, phi):
    exponent = -(t*gamma)
    a = (o*t - phi)
    return (A0*np.exp(exponent)*np.cos(a))

def sin(t,o,p):
    return np.sin(p + o*t)

def demo():
    amplitude = 1
    dampingFactor = 0.1
    gravity = 9.81616
    length = 0.25
    o0 = np.sqrt(gravity/length)
    o = np.sqrt(o0**2 - dampingFactor**2)
    angle = 0
    x = np.linspace(0,25,10000)
    plt.figure(figsize=[10,5])
    plt.plot(x,physicalPendulum(x, amplitude, dampingFactor, o, angle))
    plt.show()