import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


np.random.seed(0)
x = np.linspace(0, 10, 100).reshape(-1, 1)
y_true = np.sin(x)
noise = np.random.normal(0, 0.2, size=y_true.shape)
y = y_true + noise


model = LinearRegression()
model.fit(x, y)
y_pred = model.predict(x)


plt.figure(figsize=(10, 6))
plt.scatter(x, y, color='blue', alpha=0.5, label='sin(x) + zaj')
plt.plot(x, y_pred, color='red', linewidth=2, label='Lineáris regresszió')
plt.legend()
plt.xlabel('x')
plt.ylabel('y')
plt.grid(True)
plt.savefig("sinx_linreg.pdf", format="pdf", bbox_inches="tight")
plt.show()

residuals = y - y_pred
ss_res = np.sum(residuals**2)
ss_tot = np.sum((y - np.mean(y))**2)
r_squared = 1 - (ss_res / ss_tot)
mse = np.mean(residuals**2)
slope = model.coef_[0][0]
intercept = model.intercept_[0]
print(f"Egyenes egyenlete: y = {slope:.4f}x + {intercept:.4f}")
std_dev = np.std(residuals)
print(f"Model teljesítmény (R-négyzet): {r_squared:.4f}")
print(f"Maradékok szórása: {std_dev:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
