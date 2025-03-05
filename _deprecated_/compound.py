import matplotlib.pyplot as plt
import numpy as np
import scipy
from modules.fitToData import fit
from modules.errorProp import *

# Constant function
def f(t, c): return c + t * 0

def comp():
    # Find I0
    ballm = 0.109 # pm 0.2
    ballOffset = np.sqrt((1.000-0.030/2)**2 + (0.030/2)**2)
    rodm = 0.1116 # pm 0.2g
    rodL = 1.000 # pm 4mm
    rodOffset = 0.0095 # pm 0.2mm
    rodLinDens = rodm/rodL
    I0 = ballm*ballOffset**2 + 1/3*rodLinDens*(rodL+rodOffset)*(rodL+rodOffset)**2 - 1/3*rodLinDens*rodOffset*rodOffset**2 # CORRECT
    I0std = 0.001 # NEEDS ERROR PROP
    print(f'I0: {I0}')
    
    # Find ro+
    r0 = (rodL/2 * rodm + (ballOffset)*ballm)/(rodm+ballm)+0.043 # ESTIMATION
    r0std = 0.001 # NEEDS ERROR PROP
    print(f'ro+: {r0}')
    
    # Fit g values
    g = []
    
    g.append(fit("compoundType2Dat1", r0, lstd=False, trackingErr=0.05, phaseGuess=np.pi/1, cut=4000, cameraRate=29.97, videoRate=240, focalLength=(24*1920)/8, doPlot=True, I0=I0, r0=r0, r0Std=r0std, I0Std=I0std, m=(rodm+ballm)))

    # Plot Final Data
    x = np.linspace(0, len(g) - 1, len(g))
    z = np.linspace(0, len(g) - 1, 1000)
    g = np.array(g)

    optimal, covariance = scipy.optimize.curve_fit(f, x, g.T[0], p0=[9.81], sigma=g.T[1], absolute_sigma=True, maxfev=1*10**9)
    gstd = np.sqrt(np.diag(covariance))[0]

    print(f'Compound g: {optimal[0]:.5f} +- {gstd:.6f} ms^-2')

    return g