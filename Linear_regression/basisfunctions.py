import numpy as np
import matplotlib.pyplot as plt


def polynomial_basis(x, degree):
    return x ** degree


def gaussian_basis(x, mu, sigma):
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoidal_basis(x, mu, sigma):
    return sigmoid((x - mu) / sigma)


x_poly = np.linspace(-1, 1, 100)
x_gauss = np.linspace(-5, 5, 100)
x_sigmoid = np.linspace(-10, 10, 100)

degrees = [1, 2, 3, 4, 5]
mus = [-2, 1, 0, 1, 2]
sigma = 1.0


fig, axs = plt.subplots(1, 3, figsize=(18, 6))


for d in degrees:
    axs[0].plot(x_poly, polynomial_basis(x_poly, d), label=f'Fok {d}')
axs[0].set_title('Polinom bázisfüggvény')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')
axs[0].legend()
axs[0].grid(True)


for mu in mus:
    axs[1].plot(x_gauss, gaussian_basis(
        x_gauss, mu, sigma), label=f'μ={mu}, s={sigma}')
axs[1].set_title('Gauss bázisfüggvény')
axs[1].set_xlabel('x')
axs[1].set_ylabel('y')
axs[1].legend()
axs[1].grid(True)

for mu in mus:
    axs[2].plot(x_sigmoid, sigmoidal_basis(
        x_sigmoid, mu, sigma), label=f'μ={mu}, s={sigma}')
axs[2].set_title('Sigmoid bázisfüggvény')
axs[2].set_xlabel('x')
axs[2].set_ylabel('y')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.savefig("basisfunctions_plot.pdf", format="pdf", bbox_inches="tight")
plt.show()
