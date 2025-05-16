from sklearn.linear_model import lasso_path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('diabetes.csv')
df = df.dropna()
df.drop(columns=['Outcome'], inplace=True)

X = df.drop(columns=['DiabetesPedigreeFunction'])
y = df['DiabetesPedigreeFunction']
feature_names = X.columns

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_scaled, y)


scores = cross_val_score(lasso, X_scaled, y, cv=5,
                         scoring='neg_mean_squared_error')

print("Optimális alfa:", lasso.alpha_)
print("Keresztvalidációs MSE-ek:", -scores)
print("Átlagos MSE:", -np.mean(scores))


plt.figure(figsize=(10, 6))
plt.barh(feature_names, lasso.coef_)
plt.xlabel("Lasso-együttható érték")
plt.axvline(0, color='grey', linestyle='--')
plt.tight_layout()
plt.show()


alphas_lasso, coefs_lasso, _ = lasso_path(X_scaled, y)

plt.figure(figsize=(10, 6))
for i in range(coefs_lasso.shape[0]):
    plt.plot(alphas_lasso, coefs_lasso[i], label=feature_names[i])
plt.xscale('log')
plt.xlabel(r"$\lambda$ (log skála)")
plt.ylabel('Együtthatók értékei')
plt.legend(loc='best')
plt.grid(True)
plt.tight_layout()
plt.show()
