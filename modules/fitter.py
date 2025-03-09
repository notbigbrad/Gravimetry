import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
from modules.camera_processing import true_position, angle
from modules.error_propagation import evaluation_with_error, sp
from modules.modelling import simple_under_damped_pendulum_solution, linear_function, ode_callable_über_wrapper
from modules.Enums_and_constants import Experiment, Dependence
from modules.plotter import plot_now, do_plot_go

def simple_solution_fitting(time, subtended_angle, effective_length):

    bounds = [[0.0001, 0.0001, 0.1, -np.pi], [5, 1, 10, np.pi]]

    simple_optimal, simple_covariance_matrix = scipy.optimize.curve_fit(
        simple_under_damped_pendulum_solution,
        time,
        subtended_angle,
        p0=[0.03, 0.1, np.sqrt(9.81 / effective_length), np.pi / 4],  # <-- initial guesses : A0, gamma, omega, phi
        bounds=bounds,
        absolute_sigma=False,
        maxfev=1E9
    )

    # fitted_curve = simple_under_damped_pendulum_solution(time, *simple_optimal)
    #
    # # Plot results
    # plt.figure(figsize=(8, 5))
    # plt.scatter(time, subtended_angle, label="Data", color="red", s=10)
    # plt.plot(time, fitted_curve, label="Fitted Curve", color="blue")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Subtended Angle (rad)")
    # plt.legend()
    # plt.title("Pendulum Motion Fit")
    # plt.grid()
    # plt.show()


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



    bounds = [[-np.pi,0,0.0001,6],[np.pi,4,3,11]]

    ode_optimal, ode_covariance_matrix = scipy.optimize.curve_fit(
        lambda t, θ_initial, ω, b, g: ode_callable_über_wrapper(
        t, θ_initial, ω, b, g,
        I_given=I,
        m_given=m,
        r_o_given=r_o,
        ),
        time,
        subtended_angle,
        p0=[0.04, 1, 0.02, 9] ,     # < --- initial guesses : θ_initial, ω_initial, b, g
        bounds=bounds
    )

    # fitted_curve = ode_callable_über_wrapper(
    #     time, *ode_optimal, I_given=I, m_given=m, r_o_given=r_o
    # )
    #
    # # Plot results
    # plt.figure(figsize=(8, 5))
    # plt.scatter(time, subtended_angle, label="Data", color="red", s=10)
    # plt.plot(time, fitted_curve, label="Fitted Curve", color="blue")
    # plt.xlabel("Time (s)")
    # plt.ylabel("Subtended Angle (rad)")
    # plt.legend()
    # plt.title("ODE Model Fit to Pendulum Data")
    # plt.grid()
    # plt.show()

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


    # ----------- PARAMETER BUILDER  -----------

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
        radius_centre_of_mass
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


    return [simple_parameters, simple_covariance_matrix], [ode_parameters, ode_covariance_matrix],  [radius_centre_of_mass, radius_centre_of_mass_standard_deviation]

def compound_pendulum(p, filename, do_plot=False):
    # ----------- PRE-PROCESSING  -----------

    time, raw_x, raw_y = np.loadtxt(f'../data/{filename}.csv', delimiter=",", encoding="utf-8-sig")[
                         p['slice_bounds'][0]:p['slice_bounds'][1]].T
    time = np.linspace(0, max(time) - min(time), len(time)) / (p['capture_rate'] / p['playback_rate'])


    l_r, Δp, d_b, t_r = sp.symbols('l_r Δp d_b t_r')
    expr_ball_offset = sp.sqrt((l_r + Δp) ** 2 + ((d_b + t_r)/ 2) ** 2)

    ball_radius, ball_radius_variance = evaluation_with_error(
        my_function=expr_ball_offset,
        rod_length=[p['rod_length'], Dependence.INDEPENDENT, l_r],
        distance_to_pivot=[p['distance_to_pivot'], Dependence.INDEPENDENT, Δp],
        ball_diameter=[p['ball_diameter'], Dependence.INDEPENDENT, d_b],
        rod_thickness=[p['rod_thickness'], Dependence.INDEPENDENT, t_r]
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


    if p['ball_mass'][0] != 0:
        k_factor = (p['ball_mass'][0]) / (2 * (p['ball_mass'][0] + p['rod_mass'][0]))
        subtended_angle = np.asin ( k_factor * (raw_x - (p['ball_diameter'][0] + p['rod_thickness'][0]))  / radius_centre_of_mass)
    else:
        subtended_angle = np.asin ((raw_x / 2)/radius_centre_of_mass)

    subtended_angle -= np.mean(subtended_angle)

    # ----------- FITTING & RESIDUALS -----------

    # 1. ---------- Simple Solution Fitter -----------

    simple_parameters, simple_covariance_matrix, simple_residuals = simple_solution_fitting(
        time,
        subtended_angle,
        radius_centre_of_mass
    )

    # 2. ---------- Coupled ODE Solution Fitter -----------

    ode_parameters, ode_covariance_matrix, ode_residuals = ode_solution_fitting(
        time,
        subtended_angle,
        moment_of_inertia,
        p['ball_mass'][0]+p['rod_mass'][0],
        radius_centre_of_mass
    )

    # ----------- PLOTTING (OPTIONAL) -----------

    if do_plot:
        do_plot_go(filename, time, subtended_angle, simple_parameters, radius_centre_of_mass, simple_residuals,
                   simple_under_damped_pendulum_solution)

    return [simple_parameters, simple_covariance_matrix], [ode_parameters, ode_covariance_matrix], [radius_centre_of_mass, radius_centre_of_mass_standard_deviation]

def fitting_dataset(filename, parameters, do_plot=False):


    #----------- METHODOLOGY PROCESSING -----------

    if parameters['method'] == Experiment.DOUBLE_STRING:
        simple, differential_equation, radius_centre_of_mass = double_string_pendulum(parameters, filename, do_plot)



    elif parameters['method'] == Experiment.COMPOUND_PENDULUM:
      simple, differential_equation, radius_centre_of_mass = compound_pendulum(parameters, filename, do_plot)

    else:
        quit()


    # ----------- PARAMETER EXTRACTION FOR G -----------

    gamma = simple[0][1]
    omega = simple[0][2]

    ω, γ, r_o= sp.symbols('ω γ r_o')
    expr_g = (ω ** 2 + γ ** 2) * r_o

    g_simple, g_simple_variance = evaluation_with_error(
        my_function=expr_g,
        covariance_matrix=simple[1][1:3, 1:3],
        omega=[(omega, None), Dependence.COVARIANT, ω],
        gamma=[(gamma, None), Dependence.COVARIANT, γ],
        radius_centre_of_mass=[radius_centre_of_mass, Dependence.INDEPENDENT, r_o]
    )

    g_simple_standard_deviation = np.sqrt(g_simple_variance)

    g_differential = differential_equation[0][3]

    g_differential_standard_deviation = np.sqrt(differential_equation[1][3,3])

    return {'simple': (g_simple, g_simple_standard_deviation), 'differential_equation': (g_differential, g_differential_standard_deviation)}


def fitting_g(g):
    # Build the unsorted arrays from the dictionary
    labels = list(g.keys())
    x = np.arange(len(g))
    g_simple_values = np.array([g[label]['simple'] for label in labels])
    g_differential_values = np.array([g[label]['differential_equation'] for label in labels])

    optimal_simple, covariance_simple = scipy.optimize.curve_fit(
        linear_function, x, g_simple_values[:, 0], p0=[9.81], sigma=g_simple_values[:, 1],
        absolute_sigma=True, maxfev=10 ** 9
    )
    optimal_differential, covariance_differential = scipy.optimize.curve_fit(
        linear_function, x, g_differential_values[:, 0], p0=[9.81], sigma=g_differential_values[:, 1],
        absolute_sigma=True, maxfev=10 ** 9
    )

    g_simple_fit = optimal_simple[0]
    g_diff_eq_fit = optimal_differential[0]
    g_err_simple = np.sqrt(np.diag(covariance_simple))[0] / np.sqrt(len(g))
    g_err_diff_eq = np.sqrt(np.diag(covariance_differential))[0] / np.sqrt(len(g))

    # Plot the results using the sorted order
    plot_now(x, g_simple_values, g_differential_values, g_simple_fit, g_diff_eq_fit, g_err_simple, g_err_diff_eq,
             labels)

    print(f'FINAL g VALUES')
    print(f'g (Simple): {g_simple_fit:.5f} ± {g_err_simple:.5f} m/s²')
    print(f'g (Differential Equation): {g_diff_eq_fit:.5f} ± {g_err_diff_eq:.5f} m/s²')


