import os
import re
import numpy as np
from modules.fitter import fitting_dataset, fitting_g, Experiment


# ------ SETUP ---------
fileNames = os.listdir('../data/')
REGEX_PATTERN = r"^set\d+\.csv$"

g = []

DO_PLOT = True


# ---PARAMETER CONFIG---

# sets 1,2 performed on 23/02/25
# sets 3,4,5 performed on 25/02/25

#  SET 3 IS METAL rod 1

# SET 5 is Wooden rod

params_dict = {
    'set1': {
        'hypotenuse': (2.010, 0.5E-3),
        'horizontal': (0.485, 0.5E-3),
        'ball_diameter': (30E-3, 0.03E-3),
        'method': Experiment.DOUBLE_STRING,
    },
    'set2': {
        'hypotenuse': (2.010, 0.5E-3),
        'horizontal': (0.485, 0.5E-3),
        'ball_diameter': (30E-3, 0.03E-3),
        'method': Experiment.DOUBLE_STRING,
    },
    'set3': {
        'rod_length': (1, 0.5E-3),
        'distance_to_pivot': (4E-3, 0.02E-2),
        'rod_thickness': (1.8E-3, 0.02E-2),
        'rod_mass': (142.8E-3, 0.2E-3),
        'ball_diameter': (30E-3, 0.03E-3),
        'ball_mass': (109.0E-3, 0.2E-3),
        'method': Experiment.COMPOUND_PENDULUM,
    },
    'set4': {
        'rod_length': (1, 0.5E-3),
        'distance_to_pivot': (4E-3, 0.02E-2),
        'rod_thickness': (1.8E-3, 0.02E-2),
        'rod_mass': (142.8E-3, 0.2E-3),
        'ball_diameter': (30E-3, 0.03E-3),
        'ball_mass': (109.0E-3, 0.2E-3),
        'method': Experiment.COMPOUND_PENDULUM,
    },
    'set5': {
        'rod_length': (1, 4E-3),
        'distance_to_pivot': (9.5E-3, 0.02E-2),
        'rod_thickness': (5.4E-3, 0.02E-2),
        'rod_mass': (111.6E-3, 0.2E-3),
        'ball_diameter': (30E-3, 0.03E-3),
        'ball_mass': (109.0E-3, 0.2E-3),
        'method': Experiment.COMPOUND_PENDULUM,
    }
}



# ---- DATASETS ----

for filename in fileNames:
    if re.match(REGEX_PATTERN, filename):
        filename = filename.replace('.csv', '')
        g.append(fitting_dataset(filename,
                                 params_dict[f'{filename}'],
                                 0.05,
                                 np.pi / 2,
                                 0,
                                 do_plot=DO_PLOT))

# ---- FINAL FIT ----
fitting_g(g)



