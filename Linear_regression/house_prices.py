import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
import pandas as pd


df = pd.read_csv("house_price_regression_dataset.csv")


df = df.sample(len(df) - 900, random_state=42)


area = df["Square_Footage"]
price = df["House_Price"]


sorted_indices = np.argsort(area)
area = area.iloc[sorted_indices]
price = price.iloc[sorted_indices]


unique_area, unique_indices = np.unique(area, return_index=True)
area = area.iloc[unique_indices]
price = price.iloc[unique_indices]

spline_real = CubicSpline(area, price)
area_fine = np.linspace(area.min(), area.max(), 200)


plt.figure(figsize=(12, 6))
plt.scatter(area, price, label="Házárak", color="blue")

plt.xlabel("Méret (Négyzetláb)")
plt.ylabel("Ár (Millió USD)")
plt.title("Ház méret az ár ellenében")
plt.legend()
plt.grid()

plt.show()
