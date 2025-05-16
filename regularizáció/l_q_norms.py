import numpy as np
import matplotlib.pyplot as plt

q_values = [4, 2, 1, 0.5]
colors = ['red'] * len(q_values)

fig, axs = plt.subplots(1, len(q_values), figsize=(15, 3))
x = np.linspace(-1.5, 1.5, 400)


for ax, q, color in zip(axs, q_values, colors):
    X, Y = np.meshgrid(x, x)
    Z = np.abs(X)**q + np.abs(Y)**q
    ax.contour(X, Y, Z, levels=[1], colors=color)
    ax.set_title(f'$q = {q}$', fontsize=12)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()
