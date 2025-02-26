import numpy as np
import sympy as sp

def independent_errors(my_function, quantity, **kwargs):

    summation = 0

    variable_names = my_function.__code__.co_varnames
    variables = sp.symbols(variable_names)
    my_function_expression = my_function(*variables)

    for i, (variable_name, (value, parameter_error)) in enumerate(kwargs.items()):
        partial_derivative = sp.diff(my_function_expression, variables[i])

        input_values = [val[0] for val in kwargs.values()]
        evaluated_partial_derivative = partial_derivative.subs((dict(zip(variable_names, input_values))))

        summation += (evaluated_partial_derivative * parameter_error)**2

    return np.sqrt(summation)