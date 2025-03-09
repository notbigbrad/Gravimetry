import numpy as np
import sympy as sp

def evaluation_with_error(my_function, covariance_matrix=None, **kwargs):

    input_values = {val[1]: val[0][0] for key, val in kwargs.items()}

    variables = {key: val[1] for key, val in kwargs.items()}
    my_expression = my_function.subs(variables)

    evaluation = np.float64(my_expression.subs(input_values))
    variance = np.float64(0)

    for key, (value_tuple, symbol) in kwargs.items():
        value, parameter_error = value_tuple

        independent_partial_derivative = sp.diff(my_expression, symbol)
        independent_evaluated_partial_derivative = independent_partial_derivative.subs(input_values)
        variance += np.float64((independent_evaluated_partial_derivative * parameter_error) ** 2)

    return np.float64(evaluation), np.sqrt(np.float64(variance))