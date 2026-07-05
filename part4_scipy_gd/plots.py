"""
Part 4: Matplotlib visualizations required by the rubric:
  1) How m and b change over iterations
  2) How the error (MSE) changes over iterations

Run this after gradient_descent.py to produce both PNGs.
"""

import matplotlib.pyplot as plt
from gradient_descent import run_gradient_descent


def plot_parameter_trajectory(history, save_path="parameter_trajectory.png"):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(history["iteration"], history["m1"], marker="o", markersize=3,
            label="m1", color="#2563eb")
    ax.plot(history["iteration"], history["m2"], marker="o", markersize=3,
            label="m2", color="#16a34a")
    ax.plot(history["iteration"], history["b1"], marker="o", markersize=3,
            label="b1", color="#dc2626")
    ax.plot(history["iteration"], history["b2"], marker="o", markersize=3,
            label="b2", color="#ea580c")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Parameter value")
    ax.set_title("Parameter Trajectory: m1, m2, b1, b2 over Gradient Descent Iterations")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


def plot_error_trajectory(history, save_path="error_trajectory.png"):
    fig, ax = plt.subplots(figsize=(9, 5.5))

    ax.plot(history["iteration"], history["mse"], marker="o", markersize=3,
            color="#7c3aed")

    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean Squared Error")
    ax.set_title("MSE Trajectory over Gradient Descent Iterations")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {save_path}")


if __name__ == "__main__":
    history, final_m, final_b = run_gradient_descent(
        learning_rate=0.01, n_iterations=50, verbose=False
    )
    plot_parameter_trajectory(history)
    plot_error_trajectory(history)