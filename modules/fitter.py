import numpy as np
import scipy.optimize
from modules.error_propagation import variance_propagation
from modules.modelling import under_damped_pendulum as model, linear_function
from modules.Enums import Experiment, Dependence
from modules.plotter import plot_now, do_plot_go




def double_string_pendulum(p):

    effective_length = np.sqrt(  (p['hypotenuse'][0])**2 - (p['horizontal'][0]/2)**2  )
    effective_length_variance = variance_propagation(my_function= lambda h, d: (h ** 2 - (d / 2) ** 2) ** (1 / 2),
                                                     hypotenuse= [p['hypotenuse'], Dependence.INDEPENDENT],
                                                     horizontal= [p['horizontal'], Dependence.INDEPENDENT],)

    effective_length_standard_deviation = np.sqrt(effective_length_variance)

    effective_length_with_ball = effective_length + p['ball_diameter'][0]/2
    effective_length_with_ball_variance = variance_propagation(my_function= lambda l, d: l + d / 2,
                                                               effective_length= [(effective_length, effective_length_standard_deviation), Dependence.INDEPENDENT],
                                                               ball_diameter= [p['ball_diameter'], Dependence.INDEPENDENT])

    effective_length_with_ball_standard_deviation = np.sqrt(effective_length_with_ball_variance)

    return effective_length_with_ball, effective_length_with_ball_standard_deviation

def compound_pendulum(p):

    ball_offset = np.sqrt(  (p['rod_length'][0] - p['ball_diameter'][0]/2)**2  + (p['ball_diameter'][0]/2)**2  )
    ball_offset_variance = variance_propagation(my_function= lambda l, d: ((l - d/2) ** 2 + (d/2) ** 2) ** (1/2),
                                                rod_length= [p['rod_length'], Dependence.INDEPENDENT],
                                                ball_diameter= [p['ball_diameter'], Dependence.INDEPENDENT])

    ball_offset_standard_deviation = np.sqrt(ball_offset_variance)

    rod_linear_density = p['rod_mass'][0] / p['rod_length'][0]
    rod_linear_density_variance = variance_propagation(my_function= lambda m, l : m/l,
                                                       rod_mass= [p['rod_mass'], Dependence.INDEPENDENT],
                                                       rod_length= [p['rod_length'], Dependence.INDEPENDENT])

    rod_linear_density_standard_deviation = np.sqrt(rod_linear_density_variance)

    moment_of_inertia = (
                         (p['ball_mass'][0] * ball_offset ** 2) + (1/3 * (rod_linear_density * (p['rod_length'][0]
                        + p['distance_to_pivot'][0])) * (p['rod_length'][0]+ p['distance_to_pivot'][0]) ** 2)
                        - (1/3 * (rod_linear_density * p['distance_to_pivot'][0]) * p['distance_to_pivot'][0] ** 2)
                         )

    moment_of_inertia_variance = variance_propagation(my_function= lambda m_b, r_b, m_r, l_r, Δp, λ :
                                                        (m_b * r_b ** 2)
                                                        + (1/3 * λ * (l_r+Δp) * (l_r+Δp) ** 2)
                                                        - (1/3 * λ * Δp * Δp **2),
                                                      ball_mass= [p['ball_mass'], Dependence.INDEPENDENT],
                                                      ball_offset= [(ball_offset, ball_offset_standard_deviation), Dependence.INDEPENDENT],
                                                      rod_mass= [p['rod_mass'], Dependence.INDEPENDENT],
                                                      rod_length= [p['rod_length'], Dependence.INDEPENDENT],
                                                      distance_to_pivot= [p['distance_to_pivot'], Dependence.INDEPENDENT],
                                                      rod_linear_density= [(rod_linear_density, rod_linear_density_standard_deviation), Dependence.INDEPENDENT],)

    moment_of_inertia_standard_deviation = np.sqrt(moment_of_inertia_variance)






def fitting_dataset(filename, parameters, tracking_error=0.05, phase_guess=np.pi / 2, cut=500,
                    camera_rate=60, video_rate=60, focal_length=(24 * 1920) / 8, do_plot=False):

    # ----------- PRE-PROCESSING  -----------
    time, x, _ = np.loadtxt(f'../data/{filename}.csv', delimiter=",", encoding="utf-8-sig").T
    time = time[cut::] * (camera_rate / video_rate)
    x = x[cut::]  # -- get the data and trim it

    x = np.arctan(x / focal_length)  # focalLength (px) = focalLength (mm) * width (px) / width (mm)

    x = x - np.min(x)
    x = x - np.max(x) / 2
    x = x / np.max(x)  # Normalisation

    #----------- METHODOLOGY PROCESSING -----------

    if parameters['method'] == Experiment.DOUBLE_STRING:
        vertical_length, vertical_length_standard_deviation = double_string_pendulum(parameters)

    elif parameters['method'] == Experiment.COMPOUND_PENDULUM:
        vertical_length, vertical_length_standard_deviation = compound_pendulum(parameters)



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

    omega_naught_variance = variance_propagation(my_function= lambda ω , γ: (ω ** 2 + γ ** 2) ** (1/2),
                                                 covariance_matrix= covariance_matrix[1:3, 1:3],
                                                 omega=[(omega, None), Dependence.COVARIANT],
                                                 gamma=[(gamma, None), Dependence.COVARIANT])

    omega_naught_standard_deviation = np.sqrt(omega_naught_variance)

    g = (omega_naught ** 2) * vertical_length
    g_variance = variance_propagation(my_function= lambda ω0, l: (ω0 ** 2) * l,
                                      omega_naught=[(omega_naught, omega_naught_standard_deviation), Dependence.INDEPENDENT],
                                      length=[(vertical_length, vertical_length_standard_deviation), Dependence.INDEPENDENT])

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