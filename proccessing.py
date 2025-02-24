import os
import re
import scipy
import numpy as np
from modules.fitter import fitting_dataset, fitting_g
from modules.model import physicalPendulum as physical_pendulum


# ------ SETUP ---------
fileNames = os.listdir('./data/')
REGEX_PATTERN = r"^set\d+\.csv$"
fitting_func = scipy.optimize.curve_fit
g = []

OLD_DATA_TEST = False
DO_PLOT = False

# ---PARAMETER CONFIG---

params_dict = {
    'set1_vertical': 2.015,
    'set1_vertical_accuracy': 0.005,
    'set1_model' : physical_pendulum,

    'set2_vertical': 2.015,
    'set2_vertical_accuracy': 0.005,
    'set2_model' : physical_pendulum

    #...
}

# ---- DATASETS ----
if OLD_DATA_TEST:

    # Set known data
    l = 0.735  # compensation for ball diameter of ~40mm
    s = 0.248 / 2
    old_vertical = np.sqrt(l ** 2 - s ** 2)
    old_vertical_accuracy = 0.01

    # Initialise arrays

    g.append(fitting_dataset("bradDat1", old_vertical, old_vertical_accuracy, 0.05, np.pi / 2 + 0.4, 500, 29.97, 240, do_plot=DO_PLOT))
    g.append(fitting_dataset("bradDat2", old_vertical, old_vertical_accuracy, 0.05, np.pi / 2 + 0.4, 500, 29.97, 240, do_plot=DO_PLOT))
    g.append(fitting_dataset("matDat1", old_vertical, old_vertical_accuracy, 0.05, np.pi / 2, do_plot=DO_PLOT))
    g.append(fitting_dataset("nicoleDat1", old_vertical, old_vertical_accuracy, 0.05, np.pi / 2 - 1.8, 0, do_plot=DO_PLOT))

    # Set known data
    old_vertical = 0.737
    old_vertical_accuracy = 0.005

    g.append(fitting_dataset("hopeDat1", old_vertical, old_vertical_accuracy, 0.05, np.pi / 2, 0, do_plot=DO_PLOT))

for filename in fileNames:
    if re.match(REGEX_PATTERN, filename):
        filename = filename.replace('.csv', '')
        g.append(fitting_dataset(filename, params_dict[f'{filename}_vertical'], params_dict[f'{filename}_vertical_accuracy'],
                                 0.05, np.pi / 2, 0, model=params_dict[f'{filename}_model'], do_plot=DO_PLOT))


# ---- FINAL FIT ----
fitting_g(g)



