import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df = pd.read_csv("house_price_regression_dataset.csv")
df = df.sample(len(df) - 900, random_state=42)

area = df["Square_Footage"].values
price = df["House_Price"].values


area = np.append(area, [100, 5000])
price = np.append(price, [10**6, 2*10**5])


coefficients = np.polyfit(area, price, 1)
linear_fit = np.poly1d(coefficients)


print(
    f"Linear fit equation: y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}")

plt.figure(figsize=(12, 6))
plt.scatter(area, price, label="Házárak", color="blue")
plt.plot(area, linear_fit(area), label="Lineáris regresszió",
         color="red", linestyle="-")

for x, y in zip(area, price):
    plt.plot([x, x], [y, linear_fit(x)], color="gray",
             linestyle="--", linewidth=0.8)

plt.xlabel("Négyzetláb")
plt.ylabel("Ár (Millió USD)")
plt.legend()
plt.grid()
plt.show()


residuals = price - linear_fit(area)
ss_res = np.sum(residuals**2)
ss_tot = np.sum((price - np.mean(price))**2)
r_squared = 1 - (ss_res / ss_tot)
mse = np.mean(residuals**2)

std_dev = np.std(residuals)

print(f"Model performance (R-squared): {r_squared:.4f}")
print(f"Standard deviation of residuals: {std_dev:.4f}")
print(f"Mean Squared Error (MSE): {mse:.4f}")
print(f"The equation of the regression line is: y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}")
