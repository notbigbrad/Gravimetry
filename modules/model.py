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

def physicalODE(t, b, g, om0, thet0=np.pi/4, const=[0.2]):
    m, r, I = const

    g = 9.81  # gravity
    b = 0.01   # damping coefficient

    # Initial conditions
    y0 = [thet0, om0]

    # Time span for the solution
    t_span = (min(t), max(t))  # 0 to 10 seconds
    t_eval = np.linspace(*t_span, len(t))  # Time points for evaluation

    # Solve the ODE
    sol = scipy.integrate.solve_ivp(ode_system, t_span, y0, t_eval=t_eval, args=(b, m, g, r, I))

    # Plot the results
    plt.figure(figsize=(10, 5))
    plt.plot(sol.t, sol.y[0], label='θ (angle)')
    plt.plot(sol.t, sol.y[1], label='ω (angular velocity)', linestyle='dashed')
    plt.xlabel('Time (s)')
    plt.ylabel('Values')
    plt.legend()
    plt.title('Solution of the ODE')
    plt.grid()
    plt.show()
    return 0

# Define the ODE system
def ode_system(t, y, b, m, g, r, I, p):
    y1, y2 = y
    dydt = [y2, - (b / I) * y2 - (m * g * r / I) * np.sin(y1)]
    return dydt