import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, f1_score

# Gauss bázisfüggvény transzformáció
class GaussianFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, N=10, sigma=1.0):
        self.N = N
        self.sigma = sigma

    def fit(self, X, y=None):
        self.centers_ = np.linspace(X.min(), X.max(), self.N)
        return self

    def transform(self, X):
        X = X.ravel()[:, np.newaxis]
        C = self.centers_.ravel()
        return np.exp(-0.5 * ((X - C) / self.sigma) ** 2)

# Adatok generálása
np.random.seed(42)
n_samples = 100
X = np.linspace(0, 10, n_samples).reshape(-1, 1)
y_true = np.sin(X).ravel()
noise = np.random.normal(0, 0.2, size=n_samples)
y = y_true + noise

# Gauss paraméterek
n_centers = 20
sigma = 0.5

# Keresztvalidációs kiértékelés (egyszerű linreg)
cv = KFold(n_splits=5, shuffle=True, random_state=42)
model = make_pipeline(GaussianFeatures(N=n_centers, sigma=sigma), LinearRegression())
scores = cross_val_score(model, X, y, scoring='neg_mean_squared_error', cv=cv)
print(f"Átlagos MSE (5-szörös CV): {-np.mean(scores):.5f}")

# Szórás (standard deviation) a keresztvalidációs MSE értékekre
print(f"MSE szórása (5-szörös CV): {np.std(scores):.5f}")

# Modell illesztése és jóslás
model.fit(X, y)
y_pred = model.predict(X)

# R^2 score kiírása
r2 = r2_score(y, y_pred)
print(f"R^2 score: {r2:.5f}")

# F1 score regresszióhoz nem értelmezett, ezt jelezzük
print("F1 score: Nem értelmezhető regressziós feladathoz.")

# Ábra
plt.figure(figsize=(10, 6))
plt.scatter(X, y, label="Zajos adatok", alpha=0.6, color='gray')
plt.plot(X, y_true, label="Igazi függvény (sin)", linestyle='--', color='blue')
plt.plot(X, y_pred, label="Gauss-bázisú lineáris modell", color='red')
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True)
plt.show()
