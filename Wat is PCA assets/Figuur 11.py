import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 10,
    'axes.labelsize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 150,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})

colors = {1: '#4A90D9', 0: '#E8734A'}

np.random.seed(42)
n_fict = 300


X_benign_fict = np.column_stack([
    np.random.normal(50, 12, n_fict // 2),
    np.random.normal(1.5, 0.4, n_fict // 2)
])
X_malign_fict = np.column_stack([
    np.random.normal(50, 12, n_fict // 2),
    np.random.normal(3.5, 0.4, n_fict // 2)
])
X_fict = np.vstack([X_benign_fict, X_malign_fict])
y_fict = np.array([1] * (n_fict // 2) + [0] * (n_fict // 2))


X_fict_centered = X_fict - X_fict.mean(axis=0)
pca_fict = PCA()
X_fict_pca = pca_fict.fit_transform(X_fict_centered)

data = load_breast_cancer()
X_real = StandardScaler().fit_transform(data.data)
y_real = data.target

pca_real = PCA()
X_real_pca = pca_real.fit_transform(X_real)

fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))


ax = axes[0]
for cls, label in [(1, 'Benigne'), (0, 'Maligne')]:
    mask = y_fict == cls
    ax.hist(X_fict_pca[mask, 0], bins=30, alpha=0.55, color=colors[cls],
            label=label, density=True, edgecolor='white', linewidth=0.5)
ax.set_xlabel('Projectie op PC1 (celgrootte)')
ax.set_ylabel('Dichtheid')
ax.set_title('Sectie 3.1: Fictief — PC1 mengt klassen', fontsize=11, color='#C0392B')
ax.legend(framealpha=0.9)
ax.text(0.02, 0.95, 'Hoge variantie,\ngeen scheiding',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(facecolor='#FDEBD0', edgecolor='#E8734A', alpha=0.9, boxstyle='round,pad=0.3'))


ax = axes[1]
for cls, label in [(1, 'Benigne'), (0, 'Maligne')]:
    mask = y_real == cls
    ax.hist(X_real_pca[mask, 0], bins=30, alpha=0.55, color=colors[cls],
            label=label, density=True, edgecolor='white', linewidth=0.5)
ax.set_xlabel('Projectie op PC1')
ax.set_ylabel('Dichtheid')
ax.set_title('Sectie 4: WBCD — PC1 scheidt klassen', fontsize=11, color='#2E7D32')
ax.legend(framealpha=0.9)
ax.text(0.02, 0.95, 'Hoge variantie\nén goede scheiding',
        transform=ax.transAxes, fontsize=9, va='top',
        bbox=dict(facecolor='#D5F5E3', edgecolor='#2E7D32', alpha=0.9, boxstyle='round,pad=0.3'))

plt.suptitle('Contrast: wanneer valt variantie samen met discriminatief vermogen?', fontsize=13, y=1)
plt.tight_layout()
plt.savefig('fig_s4_contrast.png')
plt.show()
