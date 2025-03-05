import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
from compound import comp
from doubleString import ds
from modules.fitToData import fit
from modules.errorProp import *

# Constant function
def f(t, c): return c + t * 0

d = ds()
lds = len(d)
c = comp()
g = np.array(np.concatenate((c,d)))

gVal = g.T[0]
gStd = g.T[1]

x = np.linspace(0, len(g) - 1, len(g))
optimal, covariance = scipy.optimize.curve_fit(f, x, gVal, p0=[9.81], sigma=gStd, absolute_sigma=True, maxfev=1*10**9)
gstd = np.sqrt(np.diag(covariance))[0]

# Plot Final Data
z = np.linspace(0, len(gVal) - 1, 1000)
z1 = np.linspace(0, len(d) - 1, len(d))
z2 = np.linspace(len(d), len(c)-1, len(c))
g = np.array(g)

plt.figure(figsize=[10, 5])
plt.suptitle(f'g: {optimal[0]:.5f} +- {gstd:.6f} ms^-2')
plt.title(f'Measured values for g')
plt.ylabel(f'g (ms^-2)')

plt.fill_between(z1, d.T[0] - d.T[1], d.T[0] + d.T[1], color="lightcoral", alpha=0.3)
plt.errorbar(z1, d.T[0], yerr=d.T[1], color="red", marker="+", capsize=5, capthick=1, label="Double String", linewidth=0, elinewidth=1)

plt.fill_between(z2, c.T[0] - c.T[1], c.T[0] + c.T[1], color="moccasin", alpha=0.3)
plt.errorbar(z2, c.T[0], yerr=c.T[1], color="wheat", marker="+", capsize=5, capthick=1, label="Compound", linewidth=0, elinewidth=1)

plt.fill_between(z, f(z, optimal[0] - gstd), f(z, optimal[0] + gstd), color="lightskyblue", alpha=0.3)
plt.plot(z, f(z, optimal[0]), label="Least Squares Fit")

plt.plot(z, f(z, 9.81616), linestyle="--", color="green", label="Theoretical Local g")
plt.legend()
plt.show()

print(f'Final g: {optimal[0]:.5f} +- {gstd:.6f} ms^-2')
