import numpy as np
import scipy
import matplotlib.pyplot as plt

def physicalPendulum(t, A0, gamma, o, phi):
    exponent = -(t*gamma)
    a = (o*t - phi)
    return (A0*np.exp(exponent)*np.cos(a))

def sin(t,o,p):
    return np.sin(p + o*t)

def f(t, c):
    return c + t * 0

def physicalODE(t, thet0, om0, g, b, m, r, I):

    # Initial conditions
    y0 = [thet0, om0]

    # Time span for the solution
    t_span = (np.min(t), np.max(t))
    t_eval = np.linspace(*t_span, len(t))

    # Solve the ODE
    sol = scipy.integrate.solve_ivp(ode_system, t_span, y0, t_eval=t_eval, args=(b, m, g, r, I))

    # sol.y[0], sol.y[1] = angle, angular velocity
    return sol.y[0]

def ode_system(t, y, b, m, g, r, I):
    y1, y2 = y
    dydt = [y2, - (b / I) * y2 - ((m * g * r) / I) * np.sin(y1)]
    return dydt

def simpleODE(t, thet0, om0, o, b):

    # Initial conditions
    y0 = [thet0, om0]

    # Time span for the solution
    t_span = (np.min(t), np.max(t))
    t_eval = np.linspace(*t_span, len(t))

    # Solve the ODE
    sol = scipy.integrate.solve_ivp(ode_simple, t_span, y0, t_eval=t_eval, args=(b, o))

    # sol.y[0], sol.y[1] = angle, angular velocity
    return sol.y[0]

def ode_simple(t, y, b, o):
    y1, y2 = y
    dydt = [y2, - b * y2 - o * np.sin(y1)]
    return dydt