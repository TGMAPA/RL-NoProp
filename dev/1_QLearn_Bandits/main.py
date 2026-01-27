import numpy as np
import matplotlib.pyplot as plt


def compute_Q(n_iterations, values, values_probs, lr=None):
    Qk = np.zeros(n_iterations)
    r = np.random.choice(a=values, p=values_probs, size=n_iterations)

    for n in range(1, n_iterations):
        k = 1/n if lr is None else lr
        Qk[n] = Qk[n-1] + k * (r[n] - Qk[n-1])

    return Qk


def graph2Sub(qk1, qk2, title):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    final_q1 = qk1[-1]
    final_q2 = qk2[-1]

    winner_is_1 = final_q1 > final_q2

    # Slot 1
    axs[0].plot(
        qk1,
        linewidth=3 if winner_is_1 else 1.5
    )
    axs[0].axhline(final_q1, linestyle='--', alpha=0.6)
    axs[0].set_title(f"Slot 1 | Q final = {final_q1:.2f}")

    # Slot 2
    axs[1].plot(
        qk2,
        linewidth=3 if not winner_is_1 else 1.5
    )
    axs[1].axhline(final_q2, linestyle='--', alpha=0.6)
    axs[1].set_title(f"Slot 2 | Q final = {final_q2:.2f}")

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


# ===== Experimento
n_iterations = 10000
title = "10000 iterations | Slot 1 vs Slot 2 | lr = 1/k"

# Slot 1
qk1 = compute_Q(
    n_iterations,
    values=[10, 0],
    values_probs=[0.40, 0.60]
)

# Slot 2
qk2 = compute_Q(
    n_iterations,
    values=[100, 0],
    values_probs=[0.03, 0.97]
)

winner = "Slot 1 is better" if qk1[-1] > qk2[-1] else "Slot 2 is better"
graph2Sub(qk1, qk2, title + " | " + winner)
