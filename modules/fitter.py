import numpy as np
import scipy.optimize
from modules.camera_processing import true_position, angle
from modules.error_propagation import evaluation_with_error, sp
from modules.modelling import under_damped_pendulum as model, linear_function
from modules.Enums import Experiment, Dependence
from modules.plotter import plot_now, do_plot_go


def double_string_pendulum(p):

    if isinstance(p['hypotenuse'][0], list):
        p['hypotenuse'][1] = np.std(p['hypotenuse'][0])
        p['hypotenuse'][0] = np.mean(p['hypotenuse'][0])

    if isinstance(p['horizontal'][0], list):
        p['horizontal'][1] = np.std(p['horizontal'][0])
        p['horizontal'][0] = np.mean(p['horizontal'][0])

    h, d = sp.symbols('h d d_b')
    expr_length = sp.sqrt(h ** 2 - (d / 2) ** 2)

    effective_length, effective_length_variance = evaluation_with_error(
        my_function=expr_length,
        hypotenuse=[p['hypotenuse'], Dependence.INDEPENDENT, h],
        horizontal=[p['horizontal'], Dependence.INDEPENDENT, d]
    )

    effective_length_standard_deviation = np.sqrt(effective_length_variance)

    l, d = sp.symbols('l d')
    expr_length_with_ball = l + d / 2

    effective_length_with_ball, effective_length_with_ball_variance = evaluation_with_error(
        my_function=expr_length_with_ball,
        effective_length=[(effective_length, effective_length_standard_deviation), Dependence.INDEPENDENT, l],
        ball_diameter=[p['ball_diameter'], Dependence.INDEPENDENT, d]
    )

    effective_length_with_ball_standard_deviation = np.sqrt(effective_length_with_ball_variance)

    return effective_length_with_ball, effective_length_with_ball_standard_deviation


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

    # m_b, d_b, t_r, r_0, m_r = sp.symbols('m_b d_b t_r r_0 m_r')
    # expr_delta = sp.asin((m_b * (d_b + t_r)) / (2 * r_0 * (m_b + m_r)))
    #
    # delta, delta_variance = evaluation_with_error(
    #     my_function=expr_delta,
    #     ball_mass=[p['ball_mass'], Dependence.INDEPENDENT, m_b],
    #     ball_diameter=[p['ball_diameter'], Dependence.INDEPENDENT, d_b],
    #     rod_thickness=[p['rod_thickness'], Dependence.INDEPENDENT, t_r],
    #     radius_centre_of_mass=[[radius_centre_of_mass,radius_centre_of_mass_standard_deviation], Dependence.INDEPENDENT, r_0],
    #     rod_mass=[p['rod_mass'], Dependence.INDEPENDENT, m_r]
    # )
    #
    # delta_standard_deviation = np.sqrt(delta_variance)
    #
    # r_0, δ = sp.symbols('r_0 δ')
    # expr_radial_projection = r_0 * sp.cos(δ)
    #
    # radial_projection, radial_projection_variance = evaluation_with_error(
    #     my_function=expr_radial_projection,
    #     radius_centre_of_mass=[[radius_centre_of_mass,radius_centre_of_mass_standard_deviation], Dependence.INDEPENDENT, r_0],
    #     delta=[[delta, delta_standard_deviation], Dependence.INDEPENDENT, δ]
    # )
    #
    # radial_projection_standard_deviation = np.sqrt(radial_projection_variance)

    return radius_centre_of_mass, radius_centre_of_mass_standard_deviation, moment_of_inertia, moment_of_inertia_standard_deviation


def fitting_dataset(filename, parameters, tracking_error=0.05, phase_guess=np.pi / 2, cut=500,  do_plot=False):

    # ----------- PRE-PROCESSING  -----------
    time, raw_x, raw_y = np.loadtxt(f'../data/{filename}.csv', delimiter=",", encoding="utf-8-sig")[parameters['slice_bounds'][0]:parameters['slice_bounds'][1]].T
    time = np.linspace(0, max(time) - min(time), len(time)) / (parameters['capture_rate'] / parameters['playback_rate'])

    if 'z_distance' in parameters.keys():
        x, y = true_position(raw_x, raw_y, pixel_size=parameters['pixel_size'], resolution=parameters['resolution'], focal_length=parameters['focal_length'], z_distance=parameters['z_distance'])

        position_data = [time, x, y]
        zero_point = np.mean(x)


    #----------- METHODOLOGY PROCESSING -----------

    if parameters['method'] == Experiment.DOUBLE_STRING:
        vertical_length, vertical_length_standard_deviation = double_string_pendulum(parameters)

        pivot = [np.mean(x), np.mean(y + vertical_length)]
        subtended_angle = [time, angle([x,y], pivot)]

    elif parameters['method'] == Experiment.COMPOUND_PENDULUM:
        vertical_length, vertical_length_standard_deviation, moment_of_inertia, moment_of_inertia_standard_deviation = compound_pendulum(parameters)

    else:
        quit()



    bounds = [[0.99, 0.0001, 0.1, -np.pi], [1.01, 100, 10, np.pi]]
    initial_guess = [1, 0.1, np.sqrt(9.81 / vertical_length), np.pi/4] # A0, gamma, omega, phi




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
    g = np.array(g)

    optimal, covariance = scipy.optimize.curve_fit(linear_function, x, g.T[0], p0=[9.81], sigma=g.T[1],
                                                   absolute_sigma=True, maxfev=1 * 10 ** 9)
    fitted_g = optimal[0]
    g_standard_deviation = np.sqrt(np.diag(covariance))[0]
    g_standard_error = g_standard_deviation / np.sqrt(len(g))

    plot_now(x, z, linear_function, g, fitted_g, g_standard_deviation, g_standard_error)

    print(f'FINAL g VALUE FOR FIRST DATA USING MATHEMATICAL PENDULUM')
    print(f'g: {fitted_g} +- {g_standard_error} ms^-2')