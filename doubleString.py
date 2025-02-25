import matplotlib.pyplot as plt
import numpy as np
import scipy
from modules.fitToData import fit
from modules.errorProp import *

# Constant function
def f(t, c): return c + t * 0

def ds():
    # Find l
    dBall = 0.03
    dBallStd = 0.0003
    x = np.array([2.010])
    d = np.array([0.485])/2
    xstd = 0.001
    dstd = 0.001
    L = np.sqrt((x)**2 - (d)**2)
    Lstd = sqrt(L,x,d,xstd,dstd,0,"+")
    z = np.linspace(0, len(L) - 1, len(L))
    l, lcov = scipy.optimize.curve_fit(f, z, L, sigma=Lstd, absolute_sigma=True, maxfev=1*10**9)
    l = l[0]+dBall/2 # Add width of the ball
    lstd = add(1,1/2,np.sqrt(np.diag(lcov))[0],dBallStd,0)
    print(f'l: {l} +- {lstd} m')

    # Fit g values
    g = []

    g.append(fit("exp2Dat1", l, lstd, 0.05, np.pi/1, 500, 30, 30, focalLength=(24*1280)/9.8))
    g.append(fit("exp2Dat2", l, lstd, 0.05, np.pi/2+0.8, 500, 30, 30, focalLength=-2)) # camera compensation off
    g.append(fit("exp2Dat3", l, lstd, 0.05, np.pi/2+0, 500, 29.97, 30, focalLength=-2)) # camera compensation off

    # Plot Final Data
    x = np.linspace(0, len(g) - 1, len(g))
    z = np.linspace(0, len(g) - 1, 1000)
    g = np.array(g)

    optimal, covariance = scipy.optimize.curve_fit(f, x, g.T[0], p0=[9.81], sigma=g.T[1], absolute_sigma=True, maxfev=1*10**9)
    gstd = np.sqrt(np.diag(covariance))[0]

    print(f'Double string g: {optimal[0]:.5f} +- {gstd:.6f} ms^-2')

    return [optimal[0], gstd, g]