import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


np.random.seed(0)
x = np.linspace(0, 10, 100).reshape(-1, 1)
y_true = np.sin(x)
noise = np.random.normal(0, 0.2, size=y_true.shape)
y = y_true + noise


def sigmoid_basis(x, centers, scale=1.0):
    return 1 / (1 + np.exp(-(x - centers) / scale))

centers = np.linspace(0, 10, 10)  
scale = 1.0
x_sigmoid = np.hstack([sigmoid_basis(x, c, scale) for c in centers])

model = LinearRegression()
model.fit(x_sigmoid, y)
y_pred = model.predict(x_sigmoid)

# Grafikon
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', alpha=0.5, label='sin(x) + zaj')
plt.plot(x, y_pred, color='red', linewidth=2, label='Sigmoid bázisú regresszió')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.savefig("sinx_sigmoid_linreg.pdf", format="pdf", bbox_inches="tight")
plt.show()
 

residuals = y - y_pred
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)
mse = np.mean(residuals**2)
std_dev = np.std(residuals)
print(f"Model teljesítmény (R-négyzet): {r_squared:.4f}")
print(f"Maradékok szórása: {std_dev:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
