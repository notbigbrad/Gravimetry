import numpy as np
import scipy.optimize
from modules.error_propagation import variance_propagation, sp
from modules.modelling import under_damped_pendulum as model, linear_function
from modules.Enums import Experiment, Dependence
from modules.plotter import plot_now, do_plot_go


def double_string_pendulum(p):

    if isinstance(p['hypotenuse'][0], list):
        p['hypotenuse'][1] = np.std(p['hypotenuse'][0])
        p['hypotenuse'][0] = np.mean(p['hypotenuse'][0])

    effective_length = np.sqrt(p['hypotenuse'][0] ** 2 - (p['horizontal'][0] / 2) ** 2)

    h, d = sp.symbols('h d')
    expr_length = sp.sqrt(h ** 2 - (d / 2) ** 2)

    effective_length_variance = variance_propagation(
        my_function=expr_length,
        hypotenuse=[p['hypotenuse'], Dependence.INDEPENDENT, h],
        horizontal=[p['horizontal'], Dependence.INDEPENDENT, d]
    )

    effective_length_standard_deviation = np.sqrt(effective_length_variance)


    effective_length_with_ball = effective_length + p['ball_diameter'][0] / 2

    l, d = sp.symbols('l d')
    expr_length_with_ball = l + d / 2

    effective_length_with_ball_variance = variance_propagation(
        my_function=expr_length_with_ball,
        effective_length=[(effective_length, effective_length_standard_deviation), Dependence.INDEPENDENT, l],
        ball_diameter=[p['ball_diameter'], Dependence.INDEPENDENT, d]
    )

    effective_length_with_ball_standard_deviation = np.sqrt(effective_length_with_ball_variance)

    return effective_length_with_ball, effective_length_with_ball_standard_deviation


def compound_pendulum(p):

    ball_radius = np.sqrt((p['rod_length'][0] + p['distance_to_pivot'][0]) ** 2 + (p['ball_diameter'][0] / 2) ** 2)

    l, Δp, d = sp.symbols('l Δp d')
    expr_ball_offset = sp.sqrt((l + Δp) ** 2 + (d / 2) ** 2)

    ball_radius_variance = variance_propagation(
        my_function=expr_ball_offset,
        rod_length=[p['rod_length'], Dependence.INDEPENDENT, l],
        distance_to_pivot=[p['distance_to_pivot'], Dependence.INDEPENDENT, Δp],
        ball_diameter=[p['ball_diameter'], Dependence.INDEPENDENT, d]
    )

    ball_radius_standard_deviation = np.sqrt(ball_radius_variance)


    rod_linear_density = p['rod_mass'][0] / p['rod_length'][0]

    m, l = sp.symbols('m l')
    expr_rod_linear_density = m / l

    rod_linear_density_variance = variance_propagation(
        my_function=expr_rod_linear_density,
        rod_mass=[p['rod_mass'], Dependence.INDEPENDENT, m],
        rod_length=[p['rod_length'], Dependence.INDEPENDENT, l]
    )

    rod_linear_density_standard_deviation = np.sqrt(rod_linear_density_variance)


    moment_of_inertia = (
            (p['ball_mass'][0] * ball_radius ** 2) + (
                1 / 3 * (rod_linear_density * (p['rod_length'][0] + p['distance_to_pivot'][0])) * (
                    p['rod_length'][0] + p['distance_to_pivot'][0]) ** 2)
            - (1 / 3 * (rod_linear_density * p['distance_to_pivot'][0]) * p['distance_to_pivot'][0] ** 2)
    )

    m_b, r_b, λ, l_r, Δp  = sp.symbols('m_b r_b λ l_r Δp')
    expr_moment_of_inertia = ((m_b * r_b ** 2) + (1 / 3 * λ * (l_r + Δp) * (l_r + Δp) ** 2)
                              - (1 / 3 * λ * Δp * Δp ** 2))

    moment_of_inertia_variance = variance_propagation(
        my_function=expr_moment_of_inertia,
        ball_mass=[p['ball_mass'], Dependence.INDEPENDENT, m_b],
        ball_radius=[(ball_radius, ball_radius_standard_deviation), Dependence.INDEPENDENT, r_b],
        rod_linear_density=[(rod_linear_density, rod_linear_density_standard_deviation), Dependence.INDEPENDENT, λ],
        rod_length=[p['rod_length'], Dependence.INDEPENDENT, l_r],
        distance_to_pivot=[p['distance_to_pivot'], Dependence.INDEPENDENT, Δp]
    )

    moment_of_inertia_standard_deviation = np.sqrt(moment_of_inertia_variance)

    radius_centre_of_mass = np.sqrt(  (p['distance_to_pivot'][0] + p['rod_length'][0]/2) ** 2
                                      + ((p['ball_mass'][0] / (p['ball_mass'][0] + p['rod_mass'][0]))
                                      * np.sqrt(  (p['ball_diameter'][0]/2) ** 2 + (p['rod_length'][0]/2) ** 2 )) ** 2
                                      - 2 * (p['distance_to_pivot'][0] + p['rod_length'][0]/2)
                                      * ((p['ball_mass'][0] / (p['ball_mass'][0] + p['rod_mass'][0]))
                                      * np.sqrt(  (p['ball_diameter'][0]/2) ** 2 + (p['rod_length'][0]/2) ** 2 ))
                                      * np.cos( np.pi - np.arctan( p['ball_diameter'][0] / p['rod_length'][0] ) ))

    Δp, l_r, m_b, m_r, d = sp.symbols('Δp l_r m_b m_r d')
    expr_radius_centre_of_mass = sp.sqrt(
        (Δp + (l_r / 2)) ** 2 +
        ( (m_b / (m_b + m_r)) * sp.sqrt(  (d / 2) ** 2 + (l_r / 2) ** 2)  ) ** 2 -
        (2 * (Δp + (l_r / 2)) * (  (m_b / (m_b + m_r)) * sp.sqrt(  (d / 2) ** 2 + (l_r / 2) ** 2)  ) *
        sp.cos(sp.pi - sp.atan(d / l_r)))
    )

    radius_centre_of_mass_variance = variance_propagation(
        my_function=expr_radius_centre_of_mass,
        distance_to_pivot=[p['distance_to_pivot'], Dependence.INDEPENDENT, Δp],
        rod_length=[p['rod_length'], Dependence.INDEPENDENT, l_r],
        ball_mass=[p['ball_mass'], Dependence.INDEPENDENT, m_b],
        rod_mass=[p['rod_mass'], Dependence.INDEPENDENT, m_r],
        ball_diameter=[p['ball_diameter'], Dependence.INDEPENDENT, d]
    )



    radius_centre_of_mass_standard_deviation = np.sqrt(radius_centre_of_mass_variance)

    return radius_centre_of_mass, radius_centre_of_mass_standard_deviation, moment_of_inertia, moment_of_inertia_standard_deviation


def fitting_dataset(filename, parameters, tracking_error=0.05, phase_guess=np.pi / 2, cut=500,  do_plot=False,
                    camera_rate=60, video_rate=60, focal_length=(24 * 1920) / 8):

    # ----------- PRE-PROCESSING  -----------
    time, x, _ = np.loadtxt(f'../data/{filename}.csv', delimiter=",", encoding="utf-8-sig").T
    time = time - np.min(time)
    time = time[cut::] * (parameters['camera_rate']/parameters['video_rate'])
    x = x[cut::]  # -- get the data and trim it

    x = np.arctan(x / focal_length)  # focalLength (px) = focalLength (mm) * width (px) / width (mm)



    x = x - np.min(x)
    x = x - np.max(x) / 2
    x = x / np.max(x)  # Normalisation

    #----------- METHODOLOGY PROCESSING -----------

    if parameters['method'] == Experiment.DOUBLE_STRING:
        vertical_length, vertical_length_standard_deviation = double_string_pendulum(parameters)

    elif parameters['method'] == Experiment.COMPOUND_PENDULUM:
        vertical_length, vertical_length_standard_deviation, moment_of_inertia, moment_of_inertia_standard_deviation = compound_pendulum(parameters)

    else:
        quit()

    bounds = [[0.99, 0.0001, 0.1, -np.pi], [1.01, 100, 10, np.pi]]
    initial_guess = [1, 0.1, np.sqrt(9.81 / vertical_length), phase_guess] # A0, gamma, omega, phi




    #----------- FITTING & RESIDUALS -----------
    optimal, covariance_matrix = (scipy.optimize.curve_fit
                           (model, time, x,
                            p0=initial_guess, bounds=bounds, sigma=tracking_error,
                            absolute_sigma=True, maxfev=1*10**9))

    space = np.linspace(np.min(time), np.max(time), len(time))
    r = x - model(space, optimal[0], optimal[1], optimal[2], optimal[3])     # Residuals

    #----------- PLOTTING (OPTIONAL) -----------
    if do_plot:
        do_plot_go(filename, time, x, tracking_error, optimal, vertical_length, r, model)

    # ----------- PARAMETER EXTRACTION FOR G -----------

    gamma = optimal[1]
    omega = optimal[2]
    omega_naught = np.sqrt(gamma**2 + omega**2)

    ω, γ= sp.symbols('ω γ')
    expr_omega_naught = sp.sqrt(ω ** 2 + γ ** 2)

    omega_naught_variance = variance_propagation(
        my_function=expr_omega_naught,
        covariance_matrix=covariance_matrix[1:3, 1:3],
        omega=[(omega, None), Dependence.COVARIANT, ω],
        gamma=[(gamma, None), Dependence.COVARIANT, γ]
    )

    omega_naught_standard_deviation = np.sqrt(omega_naught_variance)

    if parameters['method'] == Experiment.DOUBLE_STRING:

        g = (omega_naught ** 2) * vertical_length

        ω_0, l = sp.symbols('ω_0 l')
        expr_g = (ω_0 ** 2) * l

        g_variance = variance_propagation(
            my_function=expr_g,
            omega_naught=[(omega_naught, omega_naught_standard_deviation), Dependence.INDEPENDENT, ω_0],
            length = [(vertical_length, vertical_length_standard_deviation), Dependence.INDEPENDENT, l]
        )

    elif parameters['method'] == Experiment.COMPOUND_PENDULUM:

        g = (omega_naught ** 2 * moment_of_inertia) / ((parameters['ball_mass'][0] + parameters['rod_mass'][0]) * vertical_length)

        ω_0, I, b_m, r_m, l = sp.symbols('ω_0 I b_m r_m l')
        expr_g = (ω_0 ** 2 * I) / ((b_m + r_m) * l)

        g_variance = variance_propagation(
            my_function=expr_g,
            omega_naught=[(omega_naught, omega_naught_standard_deviation), Dependence.INDEPENDENT, ω_0],
            moment_of_inertia=[(moment_of_inertia, moment_of_inertia_standard_deviation), Dependence.INDEPENDENT, I],
            ball_mass=[parameters['ball_mass'], Dependence.INDEPENDENT, b_m],
            rod_mass=[parameters['rod_mass'], Dependence.INDEPENDENT, r_m],
            length=[(vertical_length, vertical_length_standard_deviation), Dependence.INDEPENDENT, l]
        )

    g_standard_deviation = np.sqrt(g_variance)

    print(f'g: {g} +- {g_standard_deviation} ms^-2')

    return [g, g_standard_deviation]




def fitting_g(g):
    x = np.linspace(0, len(g) - 1, len(g))
    z = np.linspace(0, len(g) - 1, 1000)
    g = np.array(g)

    optimal, covariance = scipy.optimize.curve_fit(linear_function, x, g.T[0], p0=[9.81], sigma=g.T[1],
                                                   absolute_sigma=True, maxfev=1 * 10 ** 9)
    fitted_g = optimal[0]
    g_standard_deviation = np.sqrt(np.diag(covariance))[0]
    g_standard_error = g_standard_deviation / np.sqrt(len(g))

    plot_now(x, z, linear_function, g, fitted_g, g_standard_deviation, g_standard_error)

    print(f'FINAL g VALUE FOR FIRST DATA USING MATHEMATICAL PENDULUM')
    print(f'g: {fitted_g} +- {g_standard_error} ms^-2')