import os
import re
import numpy as np
from modules.fitter import fitting_dataset, fitting_g
from modules.Enums import Experiment


# ------ SETUP ---------
fileNames = os.listdir('../data/')
REGEX_PATTERN = r"^raw__.+\.csv$"
g = {}
DO_PLOT = False
# ---PARAMETER CONFIG---

# sets 1,2 performed on 23/02/25
# sets 3,4,5,6 performed on 25/02/25

#  SET 3 IS METAL rod 1

# SET 5 is Wooden rod 1


params_dict = {
    'DoublePendulum1m_1': {
        'hypotenuse': [1.016, 0.5E-3],
        'horizontal': [[0.380, 0.378, 0.379], 0.5E-3],
        'ball_diameter': (30E-3, 0.03E-3),
        'ball_mass': (109.0E-3, 0.2E-3),
        'method': Experiment.DOUBLE_STRING,

        'slice_bounds': (0, -210),
        'capture_rate': 240,
        'playback_rate': 60,
        'focal_length': 24E-3,
        'resolution': (720, 1280),
        'pixel_size': tuple(np.array([7.3, 9.8]) / [720, 1280]),
        'z_distance': 0.826
    },
    'DoublePendulum1m_2': {
        'hypotenuse': [1.016, 0.5E-3],
        'horizontal': [[0.380, 0.378, 0.379], 0.5E-3],
        'ball_diameter': (30E-3, 0.03E-3),
        'ball_mass': (109.0E-3, 0.2E-3),
        'method': Experiment.DOUBLE_STRING,

        'slice_bounds': (200, -10),
        'capture_rate': 240,
        'playback_rate': 60,
        'focal_length': 24E-3,
        'resolution': (720, 1280),
        'pixel_size': tuple(np.array([7.3, 9.8]) / [720, 1280]),
        'z_distance': 0.826
    },
    'DoublePendulum1.5m_1': {
        'hypotenuse': [1.499, 0.5E-3],
        'horizontal': [[0.375, 0.376, 0.377], 0.5E-3],
        'ball_diameter': (30E-3, 0.03E-3),
        'ball_mass': (109.0E-3, 0.2E-3),
        'method': Experiment.DOUBLE_STRING,

        'slice_bounds': (0, -220),
        'capture_rate': 240,
        'playback_rate': 60,
        'focal_length': 24E-3,
        'resolution': (720, 1280),
        'pixel_size': tuple(np.array([7.3, 9.8]) / [720, 1280]),
        'z_distance': 0.709
    },
    'DoublePendulum1.5m_2': {
        'hypotenuse': [1.499, 0.5E-3],
        'horizontal': [[0.375, 0.376, 0.377], 0.5E-3],
        'ball_diameter': (30E-3, 0.03E-3),
        'ball_mass': (109.0E-3, 0.2E-3),
        'method': Experiment.DOUBLE_STRING,

        'slice_bounds': (0, -90),
        'capture_rate': 240,
        'playback_rate': 60,
        'focal_length': 24E-3,
        'resolution': (720, 1280),
        'pixel_size': tuple(np.array([7.3, 9.8]) / [720, 1280]),
        'z_distance': 0.709
    },
    'DoublePendulum2m_1': {
        'hypotenuse': [[1.999, 1.999, 1.998], 0.5E-3],
        'horizontal': [[0.376, 0.374, 0.375], 0.5E-3],
        'ball_diameter': (30E-3, 0.03E-3),
        'ball_mass': (109.0E-3, 0.2E-3),
        'method': Experiment.DOUBLE_STRING,

        'slice_bounds': (0, -310),
        'capture_rate': 240,
        'playback_rate': 60,
        'focal_length': 24E-3,
        'resolution': (720, 1280),
        'pixel_size': tuple(np.array([7.3, 9.8]) / [720, 1280]),
        'z_distance': 0.725
    },
    'DoublePendulum2m_2': {
        'hypotenuse': [[1.999, 1.999, 1.998], 0.5E-3],
        'horizontal': [[0.376, 0.374, 0.375], 0.5E-3],
        'ball_diameter': (30E-3, 0.03E-3),
        'ball_mass': (109.0E-3, 0.2E-3),
        'method': Experiment.DOUBLE_STRING,

        'slice_bounds': (0, -250),
        'capture_rate': 240,
        'playback_rate': 60,
        'focal_length': 24E-3,
        'resolution': (720, 1280),
        'pixel_size': tuple(np.array([7.3, 9.8]) / [720, 1280]),
        'z_distance': 0.725
    },
    'MetalRod_1': {
        'rod_length': (1, 0.5E-3),
        'distance_to_pivot': (4E-3, 0.02E-3),
        'rod_thickness': (1.8E-3, 0.02E-3),
        'rod_mass': (142.8E-3, 0.2E-3),
        'ball_diameter': (0, 0),   # <-- No ball was used in this experiment
        'ball_mass': (0, 0),
        'method': Experiment.COMPOUND_PENDULUM,

        'slice_bounds': (0, -250),
        'capture_rate': 240,
        'playback_rate': 60,
        'focal_length': 24E-3,
        'resolution': (720, 1280),
        'pixel_size': tuple(np.array([7.3, 9.8]) / [720, 1280]),
        'z_distance': 0.725
    },
    'MetalRod_2': {
        'rod_length': (1, 0.5E-3),
        'distance_to_pivot': (4E-3, 0.02E-3),
        'rod_thickness': (1.8E-3, 0.02E-3),
        'rod_mass': (142.8E-3, 0.2E-3),
        'ball_diameter': (0, 0),   # <-- No ball was used in this experiment
        'ball_mass': (0, 0),
        'method': Experiment.COMPOUND_PENDULUM,

        'slice_bounds': (0, -250),
        'capture_rate': 240,
        'playback_rate': 60,
        'focal_length': 24E-3,
        'resolution': (720, 1280),
        'pixel_size': tuple(np.array([7.3, 9.8]) / [720, 1280]),
        'z_distance': 0.725/2 # < --- FIGURE THIS OUT TODO
    }
}




# ---- DATASETS ----

for filename in fileNames:
    if re.match(REGEX_PATTERN, filename):
        filename = filename.replace('.csv', '')
        g[f"{filename.replace('raw__', '')}"] = (
            fitting_dataset(filename, params_dict[f"{filename.replace('raw__', '')}"],0.05, np.pi / 2, do_plot=DO_PLOT))

# ---- FINAL FIT ----
fitting_g(g)



