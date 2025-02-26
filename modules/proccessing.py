import os
import re
import scipy
import numpy as np
from enum import Enum
from modules.fitter import fitting_dataset, fitting_g

# ------ SETUP ---------
fileNames = os.listdir('../data/')
REGEX_PATTERN = r"^set\d+\.csv$"
fitting_func = scipy.optimize.curve_fit
g = []

OLD_DATA_TEST = False
DO_PLOT = True

class Experiment(Enum):
    DOUBLE_STRING = 0
    RIGID_PENDULUM = 1


# ---PARAMETER CONFIG---

# sets 1,2 performed on 23/02/25
# sets 3,4,5 performed on 25/02/25

#  SET 3 IS METAL RULER 1

# SET 5 is Wooden ruler

params_dict = {

    'set1_hypotenuse': 0.201,
    'set1_hypotenuse_error': 0.5E-3,

    'set1_horizontal': 0.485,
    'set1_horizontal_error': 0.5E-3,

    'set1_ball_diameter': 30E-3,
    'set1_ball_diameter_error': 0.03E-3,

    'set1_method' : Experiment.DOUBLE_STRING,




    'set2_hypotenuse': 0.201,
    'set2_hypotenuse_error': 0.5E-3,

    'set2_horizontal': 0.485,
    'set2_horizontal_error': 0.5E-3,

    'set2_ball_diameter': 30E-3,
    'set2_ball_diameter_error': 0.03E-3,

    'set2_method' : Experiment.DOUBLE_STRING,




    'set3_ruler_length': 1,
    'set3_ruler_length_error': 0.5E-3,

    'set3_distance_to_pivot': 4E-3,
    'set3_distance_to_pivot_error': 0.02E-2,

    'set3_ruler_thickness': 1.8E-3,
    'set3_ruler_thickness_error': 0.02E-2,

    'set3_ruler_mass' : 142.8E-3,
    'set3_ruler_mass_error': 0.2E-3,

    'set3_ball_diameter': 30E-3,
    'set3_ball_diameter_error': 0.03E-3,

    'set3_ball_mass': 109.0E-3,
    'set3_ball_mass_error': 0.2E-3,

    'set3_method' : Experiment.RIGID_PENDULUM,



    'set4_ruler_length': 1,
    'set4_ruler_length_error': 0.5E-3,

    'set4_distance_to_pivot': 4E-3,
    'set4_distance_to_pivot_error': 0.02E-2,

    'set4_ruler_thickness': 1.8E-3,
    'set4_ruler_thickness_error': 0.02E-2,

    'set4_ruler_mass' : 142.8E-3,
    'set4_ruler_mass_error': 0.2E-3,

    'set4_ball_diameter': 30E-3,
    'set4_ball_diameter_error': 0.03E-3,

    'set4_ball_mass': 109.0E-3,
    'set4_ball_mass_error': 0.2E-3,

    'set4_method' : Experiment.RIGID_PENDULUM,



    'set5_ruler_length': 1,
    'set5_ruler_length_error': 4E-3,

    'set5_distance_to_pivot': 9.5E-3,
    'set5_distance_to_pivot_error': 0.02E-2,

    'set5_ruler_thickness': 5.4E-3,
    'set5_ruler_thickness_error': 0.02E-2,

    'set5_ruler_mass' : 111.6E-3,
    'set5_ruler_mass_error': 0.2E-3,

    'set5_ball_diameter': 30E-3,
    'set5_ball_diameter_error': 0.03E-3,

    'set5_ball_mass': 109.0E-3,
    'set5_ball_mass_error': 0.2E-3,

    'set5_method' : Experiment.RIGID_PENDULUM,






    #...


}

# ---- DATASETS ----
# if OLD_DATA_TEST:
#
#     # Set known data
#     l = 0.735  # compensation for ball diameter of ~40mm
#     s = 0.248 / 2
#     old_vertical = np.sqrt(l ** 2 - s ** 2)
#     old_vertical_accuracy = 0.01
#
#     # Initialise arrays
#
#     g.append(fitting_dataset("bradDat1", old_vertical, old_vertical_accuracy, 0.05, np.pi / 2 + 0.4, 500, 29.97, 240, do_plot=DO_PLOT))
#     g.append(fitting_dataset("bradDat2", old_vertical, old_vertical_accuracy, 0.05, np.pi / 2 + 0.4, 500, 29.97, 240, do_plot=DO_PLOT))
#     g.append(fitting_dataset("matDat1", old_vertical, old_vertical_accuracy, 0.05, np.pi / 2, do_plot=DO_PLOT))
#     g.append(fitting_dataset("nicoleDat1", old_vertical, old_vertical_accuracy, 0.05, np.pi / 2 - 1.8, 0, do_plot=DO_PLOT))
#
#     # Set known data
#     old_vertical = 0.737
#     old_vertical_accuracy = 0.005
#
#     g.append(fitting_dataset("hopeDat1", old_vertical, old_vertical_accuracy, 0.05, np.pi / 2, 0, do_plot=DO_PLOT))

for filename in fileNames:
    if re.match(REGEX_PATTERN, filename):
        filename = filename.replace('.csv', '')
        g.append(fitting_dataset(filename, params_dict,
                                 0.05, np.pi / 2,
                                 0,
                                 do_plot=DO_PLOT))


# ---- FINAL FIT ----
fitting_g(g)



