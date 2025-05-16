import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch

fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

X1 = np.array([1, 0, 0])
X2 = np.array([0, 1, 1])
X = np.column_stack((X1, X2))

y = np.array([2, 2, 3])

# Becsült beta és a projekció
beta_hat = np.linalg.inv(X.T @ X) @ X.T @ y
y_hat = X @ beta_hat
eps = y - y_hat

origin = np.zeros(3)

arrow_ratio = 0.15  

ax.quiver(*origin, *X1, color='blue', arrow_length_ratio=arrow_ratio)
ax.quiver(*origin, *X2, color='blue', arrow_length_ratio=arrow_ratio)
ax.quiver(*origin, *y, color='black', linestyle='dashed', arrow_length_ratio=arrow_ratio)
ax.quiver(*origin, *y_hat, color='forestgreen', arrow_length_ratio=arrow_ratio)
ax.plot([y[0], y_hat[0]], [y[1], y_hat[1]], [y[2], y_hat[2]],
        color='gray', linestyle='dotted', label=r'$\varepsilon = y - \hat{y}$')
ax.quiver(*origin, *(beta_hat[0]*X1), color='red', linestyle='dashed', arrow_length_ratio=arrow_ratio)
ax.quiver(*origin, *(beta_hat[1]*X2), color='red', linestyle='dashed', arrow_length_ratio=arrow_ratio)


def annotate(vec, text, offset=(0.1, 0.1, 0.1)):
    ax.text(vec[0] + offset[0], vec[1] + offset[1],
            vec[2] + offset[2], text, fontsize=12)

annotate(X1, r'$\Phi_1$', offset=(0.1, 0, 0))
annotate(X2, r'$\Phi_2$', offset=(0, 0.1, 0))
annotate(y, r'$y$', offset=(0.1, 0.1, 0.1))
annotate(y_hat, r'$\Phi\hat{\beta}$', offset=(0.1, 0.1, -0.2))
annotate(y_hat + eps, r'$y$', offset=(0.1, 0.1, 0.1)) 
annotate(beta_hat[0]*X1, r'$\Phi_1\hat{\beta}_1$', offset=(0.1, 0, 0))
annotate(beta_hat[1]*X2, r'$\Phi_2\hat{\beta}_2$', offset=(0, 0.1, 0))
annotate(y - eps / 2, r'Reziduális ($\varepsilon$)', offset=(0.1, 0.1, 0.1))
annotate((1,2,0), r"col($\Phi$)", offset=(0.0, 0.0, 0.0))


s = np.linspace(0, 3, 20)  
t = np.linspace(0, 3, 20)  
S, T = np.meshgrid(s, t)


X_plane = X1[0]*S + X2[0]*T
Y_plane = X1[1]*S + X2[1]*T
Z_plane = X1[2]*S + X2[2]*T


ax.plot_surface(X_plane, Y_plane, Z_plane, alpha=0.3, color='lightgreen')



ax.plot([y[0], y_hat[0]], [y[1], y_hat[1]], [y[2], y_hat[2]],
        color='gray', linestyle='dotted')

perp_size = 0.2  

eps_direction = eps / np.linalg.norm(eps) 
vert_start =  0.93*y_hat  
perp_size = np.linalg.norm(y_hat - vert_start)
vert_end = vert_start + perp_size * eps_direction
ax.plot([vert_start[0], vert_end[0]], [vert_start[1], vert_end[1]], [vert_start[2], vert_end[2]],
        color='black', linewidth=2)

z = vert_end-vert_start
x = y - eps

norm_req = np.linalg.norm(y_hat - vert_start)


v0 = np.cross(z, x)


v = (norm_req / np.linalg.norm(v0)) * v0

U1 = X1 / np.linalg.norm(X1)
U2 = X2 - np.dot(X2, U1) * U1
U2 = U2 / np.linalg.norm(U2)

v_horiz = v.copy()  

a = np.dot(v_horiz, U1)
b = np.dot(v_horiz, U2)

rotated_coords = np.array([b, -a])

rotated_v = rotated_coords[0]*U1 + rotated_coords[1]*U2

rotated_v = rotated_v / np.linalg.norm(rotated_v) * norm_req

offset_ref = np.array([0.1, 0.1, 0.1])

offset_ref_proj = (np.dot(offset_ref, U1))*U1 + (np.dot(offset_ref, U2))*U2

if np.dot(rotated_v, offset_ref_proj) < 0:
    rotated_v = -rotated_v


horiz_start = vert_end
horiz_end = horiz_start + rotated_v


ax.plot([horiz_start[0], horiz_end[0]], [horiz_start[1], horiz_end[1]], [horiz_start[2], horiz_end[2]],
        color='black', linewidth=2)

plane_patch = Patch(facecolor='lightgreen', alpha=0.3)


ax.set_xlim([0, 3])
ax.set_ylim([0, 3])
ax.set_zlim([0, 3])
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.tight_layout()
plt.show()