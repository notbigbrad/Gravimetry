from modules.modelling import simple_under_damped_pendulum_solution as physical_pendulum, sin, linear_function
import numpy as np
import matplotlib.pyplot as plt


def plot_now(x, g_simple_values, g_differential_values, g_simple, g_diff_eq, g_err_simple, g_err_diff_eq, labels):

    def get_sort_key(label):
        parts = label.replace("DoublePendulum", "")
        x_part, y_part = parts.split("m_")
        return (float(x_part), int(y_part))

    sorted_labels = sorted(labels, key=get_sort_key)

    sorted_indices = [labels.index(label) for label in sorted_labels]

    x_sorted = np.arange(len(sorted_labels))
    g_simple_sorted = np.array(g_simple_values)[sorted_indices]
    g_differential_sorted = np.array(g_differential_values)[sorted_indices]
    labels_sorted = [labels[i] for i in sorted_indices]

    plt.figure(figsize=(12, 6))
    plt.suptitle('Measured values for g')
    plt.ylabel('g (m/s²)')
    plt.xlabel('Experiment')

    plt.errorbar(x_sorted, g_simple_sorted[:, 0], yerr=g_simple_sorted[:, 1], fmt='o',
                 label='Simple Model', color='red', capsize=5)
    plt.errorbar(x_sorted, g_differential_sorted[:, 0], yerr=g_differential_sorted[:, 1], fmt='s',
                 label='Differential Equation Model', color='blue', capsize=5)

    plt.axhline(y=g_simple, color='red', linestyle='--',
                label=f'Fit (Simple): {g_simple:.5f} ± {g_err_simple:.1g} m/s²')
    plt.axhline(y=g_diff_eq, color='blue', linestyle='-.',
                label=f'Fit (Diff Eq): {g_diff_eq:.5f} ± {g_err_diff_eq:.1g} m/s²')
    plt.axhline(y=9.81616, color='green', linestyle='-', label='Theoretical Local g')

    plt.xticks(x_sorted, labels_sorted, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()


def do_plot_go(filename, time, subtended_angle, simple_parameters,
               simple_residuals, simple_fitted_model, ode_parameters, ode_residuals, ode_fitted_model):

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True)

    # Dense time array for smooth fitted curves.
    t_dense = np.linspace(np.min(time), np.max(time), 300)

    # ---------------------- Simple Fit Plot ----------------------
    axs[0, 0].plot(time, subtended_angle, 'ko', label='Data', markersize=4)
    axs[0, 0].plot(t_dense, simple_fitted_model(t_dense), 'b-', label='Simple Fit')

    param_text_simple = (
        "Simple Fit Parameters:\n"
        "A0 = {:.3f}\n"
        "gamma = {:.3f}\n"
        "omega = {:.3f}\n"
        "phi = {:.3f}"
    ).format(*simple_parameters)

    axs[0, 0].text(0.02, 0.98, param_text_simple, transform=axs[0, 0].transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(facecolor='white', alpha=0.7))
    axs[0, 0].set_title(f"Simple Fit (Centre of Mass Radius: {filename})")
    axs[0, 0].set_ylabel("Subtended Angle")
    axs[0, 0].legend()

    # ---------------------- ODE Fit Plot ----------------------
    axs[0, 1].plot(time, subtended_angle, 'ko', label='Data', markersize=4)
    axs[0, 1].plot(t_dense, ode_fitted_model(t_dense), 'r-', label='ODE Fit')

    param_text_ode = (
        "ODE Fit Parameters:\n"
        "θ0 = {:.3f}\n"
        "omega = {:.3f}\n"
        "b = {:.3f}\n"
        "g = {:.3f}"
    ).format(*ode_parameters)

    axs[0, 1].text(0.02, 0.98, param_text_ode, transform=axs[0, 1].transAxes,
                   fontsize=9, verticalalignment='top',
                   bbox=dict(facecolor='white', alpha=0.7))
    axs[0, 1].set_title(f"ODE Fit (Centre of Mass Radius: {filename})")
    axs[0, 1].legend()

    # ---------------------- Simple Residuals Plot ----------------------
    axs[1, 0].plot(time, simple_residuals, 'b.', label='Simple Residuals')
    axs[1, 0].axhline(0, color='k', linestyle='--')
    axs[1, 0].set_title("Simple Residuals")
    axs[1, 0].set_xlabel("Time")
    axs[1, 0].set_ylabel("Residual")
    axs[1, 0].legend()

    # ---------------------- ODE Residuals Plot ----------------------
    axs[1, 1].plot(time, ode_residuals, 'r.', label='ODE Residuals')
    axs[1, 1].axhline(0, color='k', linestyle='--')
    axs[1, 1].set_title("ODE Residuals")
    axs[1, 1].set_xlabel("Time")
    axs[1, 1].set_ylabel("Residual")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()