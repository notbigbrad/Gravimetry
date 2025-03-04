import numpy as np
import sympy as sp

def evaluation_with_error(my_function, covariance_matrix=None, **kwargs):


    input_values = {val[1]: val[0][0] for key, val in kwargs.items()}
    covariant_evaluated_partial_derivatives = []

    variables = {key: val[1] for key, val in kwargs.items()}
    my_expression = my_function.subs(variables)

    evaluation = my_expression.subs(input_values)
    variance = 0

    for key, (value_tuple, symbol) in kwargs.items():
        value, parameter_error = value_tuple


        independent_partial_derivative = sp.diff(my_expression, symbol)
        independent_evaluated_partial_derivative = independent_partial_derivative.subs(input_values)
        variance += (independent_evaluated_partial_derivative * parameter_error) ** 2

    return np.float64(evaluation), np.float64(variance)

# h, d = sp.symbols('h d')
#     expr_length = sp.sqrt(h ** 2 - (d / 2) ** 2)
#
#     effective_length, effective_length_variance = evaluation_with_error(
#         my_function=expr_length,
#         hypotenuse=[p['hypotenuse'], Dependence.INDEPENDENT, h],
#         horizontal=[p['horizontal'], Dependence.INDEPENDENT, d]
#     )
#
#     effective_length_standard_deviation = np.sqrt(effective_length_variance)
