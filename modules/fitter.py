from time import time_ns

import numpy as np
import scipy.optimize
import matplotlib.pyplot as plt
from modules.camera_processing import true_position, angle
from modules.error_propagation import evaluation_with_error, sp
from modules.modelling import simple_under_damped_pendulum_solution, linear_function, ode_callable_über_wrapper
from modules.Enums_and_constants import Experiment, Dependence
from modules.plotter import plot_now, do_plot_go

def simple_solution_fitting(time, subtended_angle, effective_length):

    bounds = [[0.0001, 0.0001, 0.1, -np.pi], [0.04, 1, 10, np.pi]]

    simple_optimal, simple_covariance_matrix = scipy.optimize.curve_fit(
        simple_under_damped_pendulum_solution,
        time,
        subtended_angle,
        p0=[0.03, 0.1, np.sqrt(9.81 / effective_length), np.pi / 4],  # <-- initial guesses : A0, gamma, omega, phi
        bounds=bounds,
        absolute_sigma=False,
        maxfev=1E9
    )

    # gamma = simple_optimal[1]
    # omega = simple_optimal[2]

    time_space = np.linspace(np.min(time), np.max(time), len(time))
    simple_residuals = subtended_angle - simple_under_damped_pendulum_solution(
        time_space,
        simple_optimal[0],
        simple_optimal[1],
        simple_optimal[2],
        simple_optimal[3]   # Residuals
    )

    return simple_optimal, simple_covariance_matrix, simple_residuals

def ode_solution_fitting(time, subtended_angle, I, m, r_o):

    θ_initial = np.pi / 4
    ω_initial = 0

    bounds = [[-np.pi,0,0,9],[np.pi,4,2,10]]

    ode_optimal, ode_covariance_matrix = scipy.optimize.curve_fit(
        lambda t, θ_initial, ω, b, g: ode_callable_über_wrapper(
        t, θ_initial, ω, b, g,
        I_given=I,
        m_given=m,
        r_o_given=r_o,
        ),
        time,
        subtended_angle,
        p0=[θ_initial, ω_initial, 0.001, 9.816] ,     # < --- initial guesses : θ_initial, ω_initial, b, g
        bounds=bounds
    )




    time_space = np.linspace(np.min(time), np.max(time), len(time))
    ode_residuals = subtended_angle - ode_callable_über_wrapper(
        time_space,
        ode_optimal[0],
        ode_optimal[1],
        ode_optimal[2],
        ode_optimal[3],
        I_given=I,
        m_given=m,
        r_o_given=r_o
    )

    return ode_optimal, ode_covariance_matrix, ode_residuals

def double_string_pendulum(p, filename, do_plot=False):

                        # ----------- PRE-PROCESSING  -----------

    time, raw_x, raw_y = np.loadtxt(f'../data/{filename}.csv', delimiter=",", encoding="utf-8-sig")[p['slice_bounds'][0]:p['slice_bounds'][1]].T
    time = np.linspace(0, max(time) - min(time), len(time)) / (p['capture_rate'] / p['playback_rate'])

    if isinstance(p['hypotenuse'][0], list):
        p['hypotenuse'][1] = np.std(p['hypotenuse'][0] / np.sqrt(len(p['hypotenuse'][0])))
        p['hypotenuse'][0] = np.mean(p['hypotenuse'][0])

    if isinstance(p['horizontal'][0], list):
        p['horizontal'][1] = np.std(p['horizontal'][0] / np.sqrt(len(p['horizontal'][0])))
        p['horizontal'][0] = np.mean(p['horizontal'][0])

    x, y = true_position(raw_x, raw_y, pixel_size=p['pixel_size'], resolution=p['resolution'], focal_length=p['focal_length'], z_distance=p['z_distance'])

    position_data = [time, x, y]
    zero_point = np.mean(x)

    # ----------- SIMPLE PENDULUM PARAMETER BUILDER  -----------

    h, d, d_b = sp.symbols('h d d_b')
    expr_length = sp.sqrt(h ** 2 - (d / 2) ** 2) + (d_b/2)

    effective_length, effective_length_variance = evaluation_with_error(
        my_function=expr_length,
        hypotenuse=[p['hypotenuse'], Dependence.INDEPENDENT, h],
        horizontal=[p['horizontal'], Dependence.INDEPENDENT, d],
        ball_diameter=[p['ball_diameter'], Dependence.INDEPENDENT, d_b],
    )

    effective_length_standard_deviation = np.sqrt(effective_length_variance)

    pivot = [np.mean(x), np.mean(y + effective_length)]
    subtended_angle = angle([x, y], pivot)

    # ----------- ODE PARAMETER BUILDER -----------

    radius_centre_of_mass, radius_centre_of_mass_standard_deviation = effective_length, effective_length_standard_deviation

    m_b, d_b, r_o = sp.symbols('m_b d_b r_o')
    expr_moment_of_inertia = (2/5 * m_b * (d_b/2)**2) + (m_b * r_o**2)

    moment_of_inertia, moment_of_inertia_variance = evaluation_with_error(
        my_function=expr_moment_of_inertia,
        ball_mass=[p['ball_mass'],Dependence.INDEPENDENT, m_b],
        ball_diameter=[p['ball_diameter'], Dependence.INDEPENDENT, d_b],
        radius_centre_of_mass=[[radius_centre_of_mass, radius_centre_of_mass_standard_deviation],Dependence.INDEPENDENT, r_o]
    )

    moment_of_inertia_standard_deviation = np.sqrt(moment_of_inertia_variance)

    # ----------- FITTING & RESIDUALS -----------

        # 1. ---------- Simple Solution Fitter -----------

    simple_parameters, simple_covariance_matrix, simple_residuals = simple_solution_fitting(
        time,
        subtended_angle,
        effective_length
    )

    # 2. ---------- Coupled ODE Solution Fitter -----------

    ode_parameters, ode_covariance_matrix, ode_residuals = ode_solution_fitting(
        time,
        subtended_angle,
        moment_of_inertia,
        p['ball_mass'][0],
        radius_centre_of_mass
    )

    # ----------- PLOTTING (OPTIONAL) -----------

    if do_plot:
        do_plot_go(filename, time, subtended_angle, simple_parameters, effective_length, simple_residuals,
                   simple_under_damped_pendulum_solution)

    print('stop')

def compound_pendulum(p):

    if p['ball_diameter'][0] != 0:

        l_r, Δp, d_b, t_r = sp.symbols('l_r Δp d_b t_r')
        expr_ball_offset = sp.sqrt((l_r + Δp) ** 2 + ((d_b + t_r)/ 2) ** 2)

        ball_radius, ball_radius_variance = evaluation_with_error(
            my_function=expr_ball_offset,
            rod_length=[p['rod_length'], Dependence.INDEPENDENT, l_r],
            distance_to_pivot=[p['distance_to_pivot'], Dependence.INDEPENDENT, Δp],
            ball_diameter=[p['ball_diameter'], Dependence.INDEPENDENT, d_b],
            rod_thickness=[p['rod_thickness'], Dependence.INDEPENDENT, t_r]
        )

    else:

        l_r, Δp = sp.symbols('l_r Δp')
        expr_ball_offset = l_r + Δp

        ball_radius, ball_radius_variance = evaluation_with_error(
            my_function=expr_ball_offset,
            rod_length=[p['rod_length'], Dependence.INDEPENDENT, l_r],
            distance_to_pivot=[p['distance_to_pivot'], Dependence.INDEPENDENT, Δp]
        )


    ball_radius_standard_deviation = np.sqrt(ball_radius_variance)

    m, l = sp.symbols('m l')
    expr_rod_linear_density = m / l

    rod_linear_density, rod_linear_density_variance = evaluation_with_error(
        my_function=expr_rod_linear_density,
        rod_mass=[p['rod_mass'], Dependence.INDEPENDENT, m],
        rod_length=[p['rod_length'], Dependence.INDEPENDENT, l]
    )

    rod_linear_density_standard_deviation = np.sqrt(rod_linear_density_variance)

    m_b, d_b, r_b, l_r, λ, Δp  = sp.symbols('m_b d_b r_b l_r λ Δp')
    expr_moment_of_inertia = (   ( (2/5 * m_b * (d_b/2) ** 2 ) + (m_b * r_b ** 2))
                                +(1 / 3 * λ * (l_r + Δp) * (l_r + Δp) ** 2)
                                -(1 / 3 * λ * Δp * Δp ** 2))

    moment_of_inertia, moment_of_inertia_variance = evaluation_with_error(
        my_function=expr_moment_of_inertia,
        ball_mass=[p['ball_mass'], Dependence.INDEPENDENT, m_b],
        ball_diameter=[p['ball_diameter'], Dependence.INDEPENDENT, d_b],
        ball_radius=[(ball_radius, ball_radius_standard_deviation), Dependence.INDEPENDENT, r_b],
        rod_length=[p['rod_length'], Dependence.INDEPENDENT, l_r],
        rod_linear_density=[(rod_linear_density, rod_linear_density_standard_deviation), Dependence.INDEPENDENT, λ],
        distance_to_pivot=[p['distance_to_pivot'], Dependence.INDEPENDENT, Δp]
    )

    moment_of_inertia_standard_deviation = np.sqrt(moment_of_inertia_variance)


    Δp, l_r, m_b, d_b, t_r, m_r = sp.symbols('Δp l_r m_b d_b t_r m_r')
    expr_radius_centre_of_mass = sp.sqrt(  ((Δp + l_r/2) ** 2)
                                           + (  (m_b ** 2 * (   (d_b + t_r) ** 2 + l_r ** 2     )   ) / (4 * (  (m_b + m_r) ** 2)   )   )
                                           + (  ( (m_b * l_r) / (m_b + m_r) ) * (Δp + l_r/2) )
                                        )

    radius_centre_of_mass, radius_centre_of_mass_variance = evaluation_with_error(
        my_function=expr_radius_centre_of_mass,
        distance_to_pivot=[p['distance_to_pivot'], Dependence.INDEPENDENT, Δp],
        rod_length=[p['rod_length'], Dependence.INDEPENDENT, l_r],
        ball_mass=[p['ball_mass'], Dependence.INDEPENDENT, m_b],
        ball_diameter=[p['ball_diameter'], Dependence.INDEPENDENT, d_b],
        rod_thickness=[p['rod_thickness'], Dependence.INDEPENDENT, t_r],
        rod_mass=[p['rod_mass'], Dependence.INDEPENDENT, m_r]
    )

    radius_centre_of_mass_standard_deviation = np.sqrt(radius_centre_of_mass_variance)

    return radius_centre_of_mass, radius_centre_of_mass_standard_deviation, moment_of_inertia, moment_of_inertia_standard_deviation


def fitting_dataset(filename, parameters, do_plot=False):


    #----------- METHODOLOGY PROCESSING -----------

    if parameters['method'] == Experiment.DOUBLE_STRING:
        vertical_length, vertical_length_standard_deviation = double_string_pendulum(parameters, filename, do_plot)



    elif parameters['method'] == Experiment.COMPOUND_PENDULUM:
        vertical_length, vertical_length_standard_deviation, moment_of_inertia, moment_of_inertia_standard_deviation = compound_pendulum(parameters)

    else:
        quit()



    bounds = [[0.0001,0.0001,0.1,-np.pi],[0.04,1,10,np.pi]]
    initial_guess = [0.03, 0.1, np.sqrt(9.81 / vertical_length), np.pi/4] # A0, gamma, omega, phi




    #----------- FITTING & RESIDUALS -----------
    optimal, covariance_matrix = (scipy.optimize.curve_fit
                           (simple_under_damped_pendulum_solution, time, subtended_angle,
                            p0=initial_guess, bounds=bounds, sigma=tracking_error,
                            absolute_sigma=True, maxfev=1*10**9))

    space = np.linspace(np.min(time), np.max(time), len(time))
    r = x - simple_under_damped_pendulum_solution(space, optimal[0], optimal[1], optimal[2], optimal[3])     # Residuals

    #----------- PLOTTING (OPTIONAL) -----------
    if do_plot:
        do_plot_go(filename, time, x, tracking_error, optimal, vertical_length, r, simple_under_damped_pendulum_solution)

    # ----------- PARAMETER EXTRACTION FOR G -----------

    gamma = optimal[1]
    omega = optimal[2]

    ω, γ= sp.symbols('ω γ')
    expr_omega_naught = sp.sqrt(ω ** 2 + γ ** 2)

    omega_naught, omega_naught_variance = evaluation_with_error(
        my_function=expr_omega_naught,
        covariance_matrix=covariance_matrix[1:3, 1:3],
        omega=[(omega, None), Dependence.COVARIANT, ω],
        gamma=[(gamma, None), Dependence.COVARIANT, γ]
    )

    omega_naught_standard_deviation = np.sqrt(omega_naught_variance)

    if parameters['method'] == Experiment.DOUBLE_STRING:


        ω_0, l = sp.symbols('ω_0 l')
        expr_g = (ω_0 ** 2) * l

        g, g_variance = evaluation_with_error(
            my_function=expr_g,
            omega_naught=[(omega_naught, omega_naught_standard_deviation), Dependence.INDEPENDENT, ω_0],
            length = [(vertical_length, vertical_length_standard_deviation), Dependence.INDEPENDENT, l]
        )

    elif parameters['method'] == Experiment.COMPOUND_PENDULUM:



        ω_0, I, b_m, r_m, l = sp.symbols('ω_0 I b_m r_m l')
        expr_g = (ω_0 ** 2 * I) / ((b_m + r_m) * l)

        g, g_variance = evaluation_with_error(
            my_function=expr_g,
            omega_naught=[(omega_naught, omega_naught_standard_deviation), Dependence.INDEPENDENT, ω_0],
            moment_of_inertia=[(moment_of_inertia, moment_of_inertia_standard_deviation), Dependence.INDEPENDENT, I],
            ball_mass=[parameters['ball_mass'], Dependence.INDEPENDENT, b_m],
            rod_mass=[parameters['rod_mass'], Dependence.INDEPENDENT, r_m],
            length=[(vertical_length, vertical_length_standard_deviation), Dependence.INDEPENDENT, l]
        )

    g_standard_deviation = np.sqrt(g_variance)

    print(f' {filename} - g: {g} +- {g_standard_deviation} ms^-2')

    return [g, g_standard_deviation]


def fitting_g(g):
    x = np.linspace(0, len(g) - 1, len(g))
    z = np.linspace(0, len(g) - 1, 1000)
    g_input = np.array([elt for elt in g.values()])
    optimal, covariance = scipy.optimize.curve_fit(linear_function, x, g_input.T[0], p0=[9.81], sigma=g_input.T[1],
                                                   absolute_sigma=True, maxfev=1 * 10 ** 9)
    fitted_g = optimal[0]
    g_standard_deviation = np.sqrt(np.diag(covariance))[0]
    g_standard_error = g_standard_deviation / np.sqrt(len(g))

    plot_now(x, z, linear_function, g, fitted_g, g_standard_deviation, g_standard_error)

    print(f'FINAL g VALUE FOR FIRST DATA USING MATHEMATICAL PENDULUM')
    print(f'g: {fitted_g} +- {g_standard_error} ms^-2')