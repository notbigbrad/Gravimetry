import multiprocessing
import scipy
from modules.modelling import *
from joblib import Parallel, delayed
from datetime import datetime as dt
import numpy as np
import matplotlib.pyplot as plt

def plot_results(time, results, errors):
    """
    Plots the results of the ODE solver with a time series plot and histogram.

    Parameters:
    - time: Time values used in the simulation.
    - results: List or array of computed results.
    - errors: Corresponding errors for each result.
    """

    results = np.array(results)
    errors = np.array(errors)

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))

    # Time Series Plot
    ax[0].plot(time, results, label='Simulated Results', alpha=0.7, color='b')
    ax[0].fill_between(time, results - errors, results + errors, color='b', alpha=0.3, label="Error Margin")
    ax[0].set_xlabel("Time")
    ax[0].set_ylabel("Subtended Angle")
    ax[0].set_title("Time Evolution of the System")
    ax[0].legend()
    ax[0].grid(True)

    # Histogram
    ax[1].hist(results, bins=30, alpha=0.7, color='r', edgecolor='black', density=True)
    ax[1].axvline(np.mean(results), color='k', linestyle='dashed', linewidth=2, label=f"Mean: {np.mean(results):.2f}")
    ax[1].set_xlabel("Result Values")
    ax[1].set_ylabel("Frequency")
    ax[1].set_title("Distribution of Results")
    ax[1].legend()
    ax[1].grid(True)

    plt.tight_layout()
    plt.show()


def ode_callable_über_wrapper(t, θ_initial, ω, b, g, I_given, m_given, r_o_given):

    def ode_fixed_params_wrapper(t, θ_initial, ω, b, g, I_given, m_given, r_o_given):

        I = I_given
        m = m_given
        r_o = r_o_given

        def physical_odes(t, y):

            θ, ω = y
            dθdt = ω
            dωdt = - (b/I) * ω - ((m*g*r_o) / I) * np.sin(θ)

            return [dθdt, dωdt]

        soln = scipy.integrate.solve_ivp(physical_odes, (t[0], t[-1]), [θ_initial, ω], t_eval=t, method="DOP853")


        return soln.y[0]

    return ode_fixed_params_wrapper(t, θ_initial, ω, b, g, I_given, m_given, r_o_given)

def prop(time, subtended_angle, constants, constants_std,  p0, n=1e3, processors=multiprocessing.cpu_count()-2, **kwargs):

    def ODE_wrapper(i, arrays, x, y):
        optimal, covariance = scipy.optimize.curve_fit(lambda t, *fittedParams: ode_callable_über_wrapper(t, *fittedParams, m_given=arrays[i][1], r_o_given=arrays[i][2], I_given=arrays[i][0]), x, y, p0=p0, maxfev=9999999)
        print(optimal[2], end='\r', flush=True)
        return optimal[2], np.sqrt(covariance[2, 2])

    print(f'Started {dt.now()}')
    
    n = int(n)
    
    noise = []
    
    for i in range(len(constants)):
        noise.append(np.random.normal(constants[i], constants_std[i], n))
    
    noise = np.array(noise).T
    
    results = []
    errors = []

    results, errors = zip(*Parallel(n_jobs=processors)(
        delayed(ODE_wrapper)(i, noise, time, subtended_angle) for i in range(len(noise))
    ))
    
    print(f'Done {dt.now()}')

    plot_results(time, results, errors)


    return results, errors, [np.mean(results), scipy.stats.sem(results)]
