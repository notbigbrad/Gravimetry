from modules.modelling import physicalPendulum as physical_pendulum, sin, f
import numpy as np
import matplotlib.pyplot as plt

def plot_now(x, z, f, g, fitted_g, g_standard_deviation, g_standard_error):
    plt.figure(figsize=[10, 5])
    plt.suptitle(f'g: {fitted_g} +- {g_standard_error} ms^-2')
    plt.title(f'Measured values for g')
    plt.ylabel(f'g (ms^-2)')
    plt.fill_between(x, g.T[0] - g.T[1], g.T[0] + g.T[1], color="lightcoral", alpha=0.3)
    plt.errorbar(x, g.T[0], yerr=g.T[1], color="red", marker="+", capsize=5, capthick=1, label="Data", linewidth=0,
                 elinewidth=1)
    plt.fill_between(z, f(z, fitted_g - g_standard_deviation), f(z, fitted_g + g_standard_deviation), color="lightskyblue", alpha=0.3)
    plt.plot(z, f(z, fitted_g), label="Least Squares Fit")
    plt.plot(z, f(z, 9.81616), linestyle="--", color="green", label="Theoretical Local g")
    plt.legend()
    plt.show()

def do_plot_go(filename, time, x, trackingErr, optimal, l, r, model):
    tSpace = np.linspace(np.min(time), np.max(time), 10000)
    plt.figure(figsize=[15, 10])
    plt.title(filename)
    plt.axis("off")
    plt.subplot(211)
    plt.suptitle("Data Plot with Fitted Model")
    plt.fill_between(time, x - trackingErr, x + trackingErr, color="lightgray")
    plt.scatter(time, x, color="red", marker="+", linewidths=1, label="Data")
    plt.plot(tSpace, model(tSpace, optimal[0], optimal[1], optimal[2], optimal[3]), label="Mathematical Pendulum Model")
    plt.plot(tSpace, sin(tSpace, np.sqrt(9.81 / l), 0), "g--", label="Theoretical")
    plt.legend()
    plt.subplot(212)
    plt.suptitle("Residual Plot")
    plt.plot(time, x * 0.05, color="lightgray")
    plt.plot(time, r, color="black")
    plt.show()