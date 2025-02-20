import os
import matplotlib.pyplot as plt
import numpy as np
import scipy
from modules.fitToData import fit

# Gather all names
fileNames = os.listdir('./data')

# Constant function
def f(t, c): return c + t * 0

# Set known data
l = 0.735  # compensation for ball diameter of ~40mm
s = 0.248 / 2
leff = np.sqrt(l ** 2 - s ** 2)
lstd = 0.01

# Initialise arrays
g = []

g.append(fit("bradDat1", leff, lstd, 0.05, np.pi/2+0.4, 500, 29.97, 240))
g.append(fit("bradDat2", leff, lstd, 0.05, np.pi/2+0.4, 500, 29.97, 240))
g.append(fit("matDat1", leff, lstd, 0.05, np.pi/2))
g.append(fit("nicoleDat1", leff, lstd, 0.05, np.pi/2-1.8, 0))

# Set known data
leff = 0.737
lstd = 0.005

g.append(fit("hopeDat1", leff, lstd, 0.05, np.pi/2, 0))

x = np.linspace(0, len(g) - 1, len(g))
z = np.linspace(0, len(g) - 1, 1000)
g = np.array(g)

optimal, covariance = scipy.optimize.curve_fit(f, x, g.T[0], p0=[9.81], sigma=g.T[1], absolute_sigma=True, maxfev=1 * 10 ** 9)
err = np.sqrt(np.diag(covariance))[0]

plt.figure(figsize=[10, 5])
plt.suptitle(f'g: {optimal[0]} +- {err} ms^-2')
plt.title(f'Measured values for g')
plt.ylabel(f'g (ms^-2)')
plt.fill_between(x, g.T[0] - g.T[1], g.T[0] + g.T[1], color="lightcoral", alpha=0.3)
plt.errorbar(x, g.T[0], yerr=g.T[1], color="red", marker="+", capsize=5, capthick=1, label="Data", linewidth=0, elinewidth=1)
plt.fill_between(z, f(z, optimal[0] - err), f(z, optimal[0] + err), color="lightskyblue", alpha=0.3)
plt.plot(z, f(z, optimal[0]), label="Least Squares Fit")
plt.plot(z, f(z, 9.81616), linestyle="--", color="green", label="Theoretical Local g")
plt.legend()
plt.show()

print(f'FINAL g VALUE FOR FIRST DATA USING MATHEMATICAL PENDULUM')
print(f'g: {optimal[0]} +- {err} ms^-2')
