import multiprocessing
import scipy
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed, parallel_backend
from datetime import datetime as dt
from scipy import stats, optimize, integrate

def plot_results(time, results, errors):
    results = np.array(results)
    errors = np.array(errors)

    if len(results) != len(time):
        if len(results) > len(time):
            results = results[:len(time)]
            errors = errors[:len(time)]
        else:
            time = time[:len(results)]

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
    ax[1].axvline(np.mean(results), color='k', linestyle='dashed', linewidth=2,
                  label=f"Mean: {np.mean(results):.2f}")
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
            dωdt = - (b / I) * ω - ((m * g * r_o) / I) * np.sin(θ)
            return [dθdt, dωdt]

        soln = integrate.solve_ivp(
            physical_odes, (t[0], t[-1]), [θ_initial, ω], t_eval=t, method="DOP853"
        )
        return soln.y[0]

    return ode_fixed_params_wrapper(t, θ_initial, ω, b, g, I_given, m_given, r_o_given)


def prop(time, subtended_angle, constants, constants_std, p0, n=1e3, processors=None, **kwargs):

    if processors is None:
        processors = max(1, multiprocessing.cpu_count() - 2)

    def ODE_wrapper(i, arrays, x, y):
        optimal, covariance = optimize.curve_fit(
            lambda t, *fittedParams: ode_callable_über_wrapper(
                t, *fittedParams, m_given=arrays[i][1], r_o_given=arrays[i][2], I_given=arrays[i][0]),x, y,p0=p0,maxfev=9999999)
        print(optimal[3], end='\r', flush=True)
        return optimal[3], np.sqrt(covariance[3, 3])

    print(f'Started {dt.now()}')

    n = int(n)

    noise = [np.random.normal(constants[i], constants_std[i], n) for i in range(len(constants))]
    noise = list(map(list, zip(*noise)))

    with parallel_backend("loky", n_jobs=processors):
        res = Parallel()(delayed(ODE_wrapper)(i, noise, time, subtended_angle)
                         for i in range(len(noise)))
    results, errors = zip(*res)

    print(f'Done {dt.now()}')
    return results, errors, [np.mean(results), stats.sem(results)]


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)

    time = np.linspace(0, 10, 100)         # 100 time values
    subtended_angle = np.sin(time)           # 100 observed data points

    constants = [1.0, 0.5, 0.3]              # Example constant values (I, m, r_o)
    constants_std = [0.1, 0.05, 0.03]        # Standard deviations for each constant
    p0 = [0.1, 0.1, 0.1]                     # Initial guess for curve fitting parameters

    results, errors, summary = prop(time, subtended_angle, constants, constants_std, p0)


