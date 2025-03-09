# DO NOT EDIT
# ONLY MAKE CHANGES TO COPY
import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

def simple_under_damped_pendulum_solution(t, A0, gamma, o, phi):
    exponent = -(t*gamma)
    a = (o*t - phi)
    return (A0*np.exp(exponent)*np.cos(a))

def sin(t,o,p, a):
    return a*np.sin(p + o*t)

def linear_function(t, c):
    return c + t * 0

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

        soln = scipy.integrate.solve_ivp(physical_odes, (t[0], t[-1]), [θ_initial, ω], t_eval=t, method="RK45")


        return soln.y[0]

    return ode_fixed_params_wrapper(t, θ_initial, ω, b, g, I_given, m_given, r_o_given)


