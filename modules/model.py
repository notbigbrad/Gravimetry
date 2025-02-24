# DO NOT EDIT
# ONLY MAKE CHANGES TO COPY

import numpy as np

def physicalPendulum(t, A0, gamma, o, phi):
    exponent = -(t*gamma)
    a = (o*t - phi)
    return (A0*np.exp(exponent)*np.cos(a))

def sin(t,o,p):
    return np.sin(p + o*t)

def f(t, c):
    return c + t * 0