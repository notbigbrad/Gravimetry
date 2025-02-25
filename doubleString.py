import matplotlib.pyplot as plt
import numpy as np
import scipy
from modules.fitToData import fit

# Constant function
def f(t, c): return c + t * 0

def ds():
    # Find l
    x = np.array([2.010])
    d = np.array([0.485])/2
    xvar = 0.005
    dvar = 0.005
    L = np.sqrt((x)**2 - (d)**2)
    Lvar = dvar**2*(d/L)**2+xvar**2*(x/L)**2
    print(L)
    print(Lvar)
    z = np.linspace(0, len(L) - 1, len(L))
    l, lcov = scipy.optimize.curve_fit(f, z, L, sigma=Lvar, absolute_sigma=True, maxfev=1*10**9)
    l = l[0]+0.015 # Add width of the ball
    lstd = np.sqrt(np.diag(lcov))[0] # Ignore error in ball
    print(f'l: {l} +- {lstd} m')

    # Fit g values
    g = []

    g.append(fit("exp2Dat1", l, lstd, 0.05, np.pi/1, 500, 30, 30, doPlot=True, focalLength=(24*1280)/9.8))
    g.append(fit("exp2Dat1", l, lstd, 0.05, np.pi/2+1.4, 500, 30, 30, doPlot=True, focalLength=-2)) # camera compensation off

    # Plot Final Data
    x = np.linspace(0, len(g) - 1, len(g))
    z = np.linspace(0, len(g) - 1, 1000)
    g = np.array(g)

    optimal, covariance = scipy.optimize.curve_fit(f, x, g.T[0], p0=[9.81], sigma=g.T[1], absolute_sigma=True, maxfev=1*10**9)
    gstd = np.sqrt(np.diag(covariance))[0]

    plt.figure(figsize=[10, 5])
    plt.suptitle(f'g: {optimal[0]:.5f} +- {gstd:.6f} ms^-2')
    plt.title(f'Measured values for g')
    plt.ylabel(f'g (ms^-2)')
    plt.fill_between(x, g.T[0] - g.T[1], g.T[0] + g.T[1], color="lightcoral", alpha=0.3)
    plt.errorbar(x, g.T[0], yerr=g.T[1], color="red", marker="+", capsize=5, capthick=1, label="Data", linewidth=0, elinewidth=1)
    plt.fill_between(z, f(z, optimal[0] - gstd), f(z, optimal[0] + gstd), color="lightskyblue", alpha=0.3)
    plt.plot(z, f(z, optimal[0]), label="Least Squares Fit")
    plt.plot(z, f(z, 9.81616), linestyle="--", color="green", label="Theoretical Local g")
    plt.legend()
    plt.show()

    print(f'Double string g: {optimal[0]:.5f} +- {gstd:.6f} ms^-2')

    return [optimal[0], gstd, g]