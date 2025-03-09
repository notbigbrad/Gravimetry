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

    # Set x-axis ticks and labels
    plt.xticks(x_sorted, labels_sorted, rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.show()

def do_plot_go(filename, time, x, optimal, l, r, model):

    tSpace = np.linspace(np.min(time), np.max(time), 10000)
    plt.figure(figsize=[15, 10])
    plt.title(filename)
    plt.axis("off")
    plt.subplot(211)
    plt.suptitle("Data Plot with Fitted Model")
    plt.fill_between(time, x - 0.00872665, x + 0.00872665, color="lightgray") # <--- using half a degree as error, might revise later
    plt.scatter(time, x, color="red", marker="+", linewidths=1, label="Data")
    plt.plot(tSpace, model(tSpace, optimal[0], optimal[1], optimal[2], optimal[3]), label="Mathematical Pendulum Model")
    plt.plot(tSpace, sin(tSpace, np.sqrt(9.81 / l), 0, optimal[0]), "g--", label="Theoretical")
    plt.legend()
    plt.subplot(212)
    plt.suptitle("Residual Plot")
    plt.plot(time, x * 0.05, color="lightgray")
    plt.plot(time, r, color="black")
    plt.show()