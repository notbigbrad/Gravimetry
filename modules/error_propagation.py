import numpy as np
import sympy as sp
from modules.Enums_and_constants import Experiment, Dependence


def evaluation_with_error(my_function, covariance_matrix=None, **kwargs):

    input_values = {val[2]: val[0][0] for key, val in kwargs.items()}
    covariant_evaluated_partial_derivatives = []

    variables = {key: val[2] for key, val in kwargs.items()}
    my_expression = my_function.subs(variables)

    evaluation = my_expression.subs(input_values)
    variance = 0

    for key, (value_tuple, dependence, symbol) in kwargs.items():
        value, parameter_error = value_tuple

        if dependence == Dependence.INDEPENDENT:
            independent_partial_derivative = sp.diff(my_expression, symbol)
            independent_evaluated_partial_derivative = independent_partial_derivative.subs(input_values)
            variance += (independent_evaluated_partial_derivative * parameter_error) ** 2

        elif dependence == Dependence.COVARIANT:
            covariant_evaluated_partial_derivatives.append(sp.diff(my_expression, symbol).subs(input_values))

    if covariance_matrix is not None:
        jacobian = np.array(covariant_evaluated_partial_derivatives)
        variance += (jacobian @ covariance_matrix @ jacobian.T)

    return np.float64(evaluation), np.float64(variance)
