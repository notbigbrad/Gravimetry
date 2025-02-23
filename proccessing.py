import os
import re
import numpy as np
import scipy
from modules.fitToData import fit
from modules import curve_fit
from modules import least_squares_fit
from modules.final_result_plotter import plot_now

# ------ SETUP ---------
fileNames = os.listdir('./data/')
REGEX_PATTERN = r"^set\d+\.csv$"
fitting_func = scipy.optimize.curve_fit
g = []

OLD_DATA_TEST = True
DO_PLOT = False

# Constant function
def f(t, c): return c + t * 0





# ---PARAMETER CONFIG---
vertical = 2.015
vertical_accuracy = 0.005

# ---- DATASETS ----
if OLD_DATA_TEST:
    # Set known data
    l = 0.735  # compensation for ball diameter of ~40mm
    s = 0.248 / 2
    old_vertical = np.sqrt(l ** 2 - s ** 2)
    old_vertical_accuracy = 0.01

    # Initialise arrays

    g.append(fit("bradDat1", old_vertical, old_vertical_accuracy, 0.05, np.pi / 2 + 0.4, 500, 29.97, 240))
    g.append(fit("bradDat2", old_vertical, old_vertical_accuracy, 0.05, np.pi / 2 + 0.4, 500, 29.97, 240))
    g.append(fit("matDat1", old_vertical, old_vertical_accuracy, 0.05, np.pi / 2))
    g.append(fit("nicoleDat1", old_vertical, old_vertical_accuracy, 0.05, np.pi / 2 - 1.8, 0))

    # Set known data
    old_vertical = 0.737
    old_vertical_accuracy = 0.005

    g.append(fit("hopeDat1", old_vertical, old_vertical_accuracy, 0.05, np.pi / 2, 0))

for filename in fileNames:
    if re.match(REGEX_PATTERN, filename):
        g.append(fit(filename.split(".")[0], vertical, vertical_accuracy,
                     0.05, np.pi / 2, 0, doPlot=DO_PLOT))


x = np.linspace(0, len(g) - 1, len(g))
z = np.linspace(0, len(g) - 1, 1000)
g = np.array(g)


if fitting_func is scipy.optimize.curve_fit:
    optimal, covariance = fitting_func(f, x, g.T[0], p0=[9.81], sigma=g.T[1], absolute_sigma=True, maxfev=1 * 10 ** 9)
elif fitting_func is scipy.optimize.least_squares:
    # optimal, covariance =
    pass

err = np.sqrt(np.diag(covariance))[0]
std_err = np.sqrt(np.diag(covariance)/len(g))[0]

plot_now(x, z, f, g, optimal, err)

print(f'FINAL g VALUE FOR FIRST DATA USING MATHEMATICAL PENDULUM')
print(f'g: {optimal[0]} +- {err} ms^-2')
