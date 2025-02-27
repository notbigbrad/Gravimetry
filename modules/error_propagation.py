import numpy as np
import sympy as sp
from modules.Enums import Experiment, Dependence

def variance_propagation(my_function, covariance_matrix=None, **kwargs):
    result = 0
    input_values = [val[0][0] for val in kwargs.values()]
    evaluated_covariant_partial_derivatives = []

    variable_names = my_function.__code__.co_varnames
    variables = sp.symbols(variable_names)
    my_expression = my_function(*variables)

    for i, entry in enumerate(kwargs.values()):
        entry.append(variables[i])

    for i, (variable_name, [(value, parameter_error), dependence, symbol]) in enumerate(kwargs.items()):

        if dependence == Dependence.INDEPENDENT:

            independent_partial_derivative = sp.diff(my_expression, symbol)
            independent_evaluated_partial_derivative = independent_partial_derivative.subs((dict(zip(variable_names, input_values))))
            result += (independent_evaluated_partial_derivative * parameter_error)**2

        elif dependence == Dependence.COVARIANT:

            evaluated_covariant_partial_derivatives.append(sp.diff(my_expression, symbol).subs((dict(zip(variable_names, input_values)))))

    if covariance_matrix is not None:

        jacobian = np.array(evaluated_covariant_partial_derivatives)
        result += (jacobian @ covariance_matrix @ jacobian.T)

    return np.float64(result)
