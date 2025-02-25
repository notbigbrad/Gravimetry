import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
from compound import comp
from doubleString import ds
from modules.fitToData import fit

# Constant function
def f(t, c): return c + t * 0

_, _, d = ds()
lds = len(d)
# _, _, c = comp()
c = []
g = np.concatenate(d,c)

x = np.linspace(0, len(g) - 1, len(g))

optimal, covariance = scipy.optimize.curve_fit(f, x, g.T[0], p0=[9.81], sigma=g.T[1], absolute_sigma=True, maxfev=1*10**9)
gstd = np.sqrt(np.diag(covariance))[0]

# Plot Final Data
z = np.linspace(0, len(g) - 1, 1000)
g = np.array(g)

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
