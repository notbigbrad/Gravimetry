import numpy as np
import sympy as sp
from modules.Enums import Experiment, Dependence

def variance_propagation(my_function, **kwargs):

    result = 0
    covariant_partial_derivative = []
    covariance_matrix = kwargs.pop('covariance_matrix')
    print(kwargs)


    variable_names = my_function.__code__.co_varnames
    variables = sp.symbols(variable_names)
    my_expression = my_function(*variables)

    for i, (variable_name, [(value, parameter_error), dependence]) in enumerate(kwargs.items()):
        if dependence == Dependence.INDEPENDENT:
            independent_partial_derivative = sp.diff(my_expression, variables[i])
            input_values = [val[0][0] for val in kwargs.values()]
            evaluated_partial_derivative = independent_partial_derivative.subs((dict(zip(variable_names, input_values))))
            result += (evaluated_partial_derivative * parameter_error)**2

        elif dependence == Dependence.COVARIANT:
            pass

    return np.float64(result)

def covariant_parameters():
    pass