import multiprocessing
import scipy
import os
from modules.modelling import *
from joblib import Parallel, delayed
from datetime import datetime as dt

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

def prop(time, subtended_angle, constants, constants_std,  p0, n=1e5, processors=multiprocessing.cpu_count()-2, **kwargs):

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
    
    return results, errors, [np.mean(results), scipy.stats.sem(results)]