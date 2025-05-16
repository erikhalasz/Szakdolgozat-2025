from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import ElasticNet
from sklearn.datasets import make_regression

np.random.seed(42)
X, y, coef_true = make_regression(
    n_samples=100,
    n_features=10,
    n_informative=5,
    noise=10,
    coef=True
)

params = [
    (0.1, 0.1),
    (0.1, 0.5),
    (0.1, 0.9),
    (1.0, 0.1),
    (1.0, 0.5),
    (1.0, 0.9),
    (10.0, 0.1),
    (10.0, 0.5),
    (10.0, 0.9),
]

fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharey=True)


best_score = float('inf')
best_params = None

for ax, (alpha, l1_ratio) in zip(axes.ravel(), params):
    model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio,
                       fit_intercept=False, max_iter=10000)
    model.fit(X, y)
    y_pred = model.predict(X)
    mse = mean_squared_error(y, y_pred)
    if mse < best_score:
        best_score = mse
        best_params = (alpha, l1_ratio)
    ax.bar(range(X.shape[1]), model.coef_, color='blue', alpha=0.6)
    ax.set_title(r"$\lambda_1$ = %.1f, $\lambda_2$ = %.1f" % (alpha, l1_ratio))
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel("Paraméter index")
    ax.set_ylabel("Együttható érték")

plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.show()

print(
    f"Legjobb fit: alpha={best_params[0]}, l1_ratio={best_params[1]}, MSE={best_score:.2f}")
