import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.datasets import make_regression

X, y, coef_true = make_regression(
    n_samples=100, n_features=10, noise=10.0, coef=True, random_state=42
)


alphas = np.logspace(-2, 3, 100)

coefs = []

for alpha in alphas:
    ridge = Ridge(alpha=alpha, fit_intercept=False)
    ridge.fit(X, y)
    coefs.append(ridge.coef_)

coefs = np.array(coefs)

plt.figure(figsize=(10, 6))
for i in range(X.shape[1]):
    plt.plot(alphas, coefs[:, i], label=f'Feature {i+1}')
plt.xscale('log')
plt.xlabel(r'$\lambda$ (log skála)')
plt.ylabel('Ridge Együtthatók')
plt.legend(loc='best', fontsize='small')
plt.grid(True)
plt.tight_layout()
plt.show()
