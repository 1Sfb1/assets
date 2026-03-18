import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from mpl_toolkits.mplot3d import Axes3D

np.random.seed(42)

n_samples = 3500
n_groups = 8
t_min = 1.5 * np.pi
t_max = 6.8 * np.pi
radius_scale = 0.85
height_scale = 32

t = np.sort(np.random.uniform(t_min, t_max, n_samples))

x = radius_scale * t * np.cos(t) + 0.35 * np.random.randn(n_samples)
y = np.random.uniform(0, height_scale, n_samples)
z = radius_scale * t * np.sin(t) + 0.35 * np.random.randn(n_samples)

X = np.column_stack([x, y, z])

bins = np.linspace(t.min(), t.max(), n_groups + 1)
group_labels = np.digitize(t, bins) - 1
group_labels = np.clip(group_labels, 0, n_groups - 1)

colors = ['#e6194b', '#f58231', '#ffe119', '#3cb44b', 
          '#4363d8', '#911eb4', '#f032e6', '#42d4f4']


pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

explained_var = pca.explained_variance_ratio_ * 100

fig = plt.figure(figsize=(15, 6.8), dpi=120)


ax1 = fig.add_subplot(121, projection='3d')

for g in range(n_groups):
    mask = group_labels == g
    ax1.scatter(X[mask, 0], X[mask, 1], X[mask, 2],
                c=colors[g], s=11, alpha=0.88, edgecolors='none')

ax1.set_title('Swiss Roll in $\\mathbb{R}^3$', fontsize=17, fontweight='bold', pad=20)
ax1.set_xlabel('$x$', fontsize=13, labelpad=10)
ax1.set_ylabel('$y$', fontsize=13, labelpad=10)
ax1.set_zlabel('$z$', fontsize=13, labelpad=10)

ax1.view_init(elev=18, azim=-58)
ax1.tick_params(labelsize=9)

for pane in [ax1.xaxis, ax1.yaxis, ax1.zaxis]:
    pane.pane.fill = False
    pane.pane.set_edgecolor('0.85')

ax1.grid(True, alpha=0.25)

ax2 = fig.add_subplot(122)
for g in range(n_groups):
    mask = group_labels == g
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1],
                c=colors[g], s=13, alpha=0.9, 
                edgecolor='white', linewidth=0.25, zorder=n_groups - g)

ax2.set_title('PCA Projectie op PC1 en PC2', fontsize=17, fontweight='bold', pad=20)
ax2.set_xlabel('Eerste Principale Component (PC1)', fontsize=13, labelpad=10)
ax2.set_ylabel('Tweede Principale Component (PC2)', fontsize=13, labelpad=10)
ax2.tick_params(labelsize=10)
ax2.grid(True, alpha=0.2, linestyle=':')

info_text = (
    f'PC1: {explained_var[0]:.1f}% verklaard\n'
    f'PC2: {explained_var[1]:.1f}% verklaard\n'
    f'Totaal: {explained_var[0] + explained_var[1]:.1f}%'
)

ax2.text(0.02, 0.98, info_text, transform=ax2.transAxes, fontsize=11,
         verticalalignment='top', bbox=dict(boxstyle="round,pad=0.6", 
                                           facecolor="white", edgecolor="gray", alpha=0.95))

plt.tight_layout(w_pad=4)
plt.show()
