import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Polygon

# Ábrához használt rács generálása
x = np.linspace(-3, 3, 400)
y = np.linspace(-3, 3, 400)
X, Y = np.meshgrid(x, y)

# Kvadratikus veszteségfüggvény: ellipszis kontúrokhoz
Z = (X-2/np.sqrt(8))**2 + (Y - (1+2/np.sqrt(8)))**2  # aszimmetrikus kvadratikus forma

# Kontúrszintek
levels1 = [0.5, np.sqrt(2+2/np.sqrt(2))-1.14,1]
levels2 = [0.2, 0.4, np.sqrt(2+2/np.sqrt(2))-1.14]

# Ábra beállítása
fig, axs = plt.subplots(1, 2, figsize=(10, 7))

# --- L1 regularizáció (bal oldali ábra) ---
axs[0].contour(X, Y, Z, levels=levels1, colors='blue')

# Gyémánt alak (L1 normához)
diamond = Polygon([[0, 1], [1, 0], [0, -1], [-1, 0]], closed=True, edgecolor='darkred', facecolor='red', alpha=0.2)
axs[0].add_patch(diamond)
axs[0].set_xlim(-2.9, 2.9)
axs[0].set_ylim(-2.9, 2.9)
axs[0].axhline(0, color='black', lw=1)
axs[0].axvline(0, color='black', lw=1)
axs[0].set_aspect('equal')
axs[0].axis('off')

# --- L2 regularizáció (jobb oldali ábra) ---
axs[1].contour(X, Y, Z, levels=levels2, colors='blue')

# Kör alak (L2 normához)
circle = Circle((0, 0), radius=1, edgecolor='darkred', facecolor='red', alpha=0.2)
axs[1].add_patch(circle)

axs[1].set_xlim(-2.9, 2.9)
axs[1].set_ylim(-2.9, 2.9)
axs[1].axhline(0, color='black', lw=1)
axs[1].axvline(0, color='black', lw=1)
axs[1].set_aspect('equal')
axs[1].axis('off')

# Középpont koordinátái
center_x = 2 / np.sqrt(8)
center_y = 1 + 2 / np.sqrt(8)

# Pont beszúrása
axs[0].plot(center_x, center_y, 'bo') # L1 regularizáció
axs[0].text(center_x, center_y-0.27, r'$\hat{\beta}$', fontsize=10, color='black')
axs[1].text(center_x, center_y-0.27, r'$\hat{\beta}$', fontsize=10, color='black')
axs[1].plot(center_x, center_y, 'bo')  # L2 regularizáció

axs[0].plot(0,1, 'ro')  # L1 regularizáció
axs[0].text(0, 0.675, r'$\beta^*$', fontsize=10, color='black')
y = 1/(2+2/np.sqrt(2))*(1+2/(2*np.sqrt(2)))
axs[1].plot(3.7/(4*np.sqrt(2)+4),1.85*y, 'ro') # L2 regularizáció
axs[1].text(3.8/(4*np.sqrt(2)+4), 1.9*y-0.25, r'$\beta^*$', fontsize=10, color='black')

# Ábra összeillesztése
plt.tight_layout()
plt.show()
