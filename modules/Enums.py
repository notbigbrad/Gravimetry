from enum import Enum

class Experiment(Enum):
    DOUBLE_STRING = 0
    COMPOUND_PENDULUM = 1

class Dependence(Enum):
    INDEPENDENT = 0
    COVARIANT = 1
