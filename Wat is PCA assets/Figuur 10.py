
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 11,
    'figure.dpi': 200,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
    'font.family': 'serif',
})


data = load_breast_cancer()
X = StandardScaler().fit_transform(data.data)
y = data.target  # Labels, goed aardig/kwaadaardig


pca = PCA()
X_pca = pca.fit_transform(X)
eigenvalues = pca.explained_variance_
evr = pca.explained_variance_ratio_
cum_evr = np.cumsum(evr)
k95 = np.argmax(cum_evr >= 0.95) + 1
n, d = data.data.shape


fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), gridspec_kw={'width_ratios': [1.1, 1]})

ax = axes[0]
colors = {1: "#0F63BC", 0: '#E8734A'}
labels = {1: 'Benigne', 0: 'Maligne'}
for cls in [1, 0]:
    mask = y == cls
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], c=colors[cls],
               label=labels[cls], alpha=0.6, s=28, edgecolors='white', linewidths=0.3)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('PCA-projectie: Breast Cancer Wisconsin')
ax.legend(loc='upper left', framealpha=0.9)
ax.text(0.02, 0.02,
        f'$n={n},\\ d={d}$\n$\\lambda_1={eigenvalues[0]:.1f},\\ \\lambda_2={eigenvalues[1]:.1f}$\n'
        f'$\\mathrm{{EVR}}_2 = {cum_evr[1]*100:.1f}\\%$',
        transform=ax.transAxes, fontsize=9, va='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))


ax2 = axes[1]
k_show = 10
x_pos = np.arange(1, k_show + 1)
ax2.bar(x_pos, evr[:k_show] * 100, color='#E8734A', alpha=0.7, label='Per component')
ax2.plot(x_pos, cum_evr[:k_show] * 100, 'o-', color='#2E7D32', linewidth=2, markersize=6, label='Cumulatief')
ax2.axhline(95, color='gray', linestyle='--', linewidth=1, alpha=0.6)
ax2.text(k_show + 0.1, 95.5, '95%', fontsize=9, color='gray')
ax2.annotate(f'$k = {k95}$: EVR $\\approx$ {cum_evr[k95-1]*100:.0f}%',
             xy=(k95, cum_evr[k95-1]*100), xytext=(k95 + 1.5, cum_evr[k95-1]*100 - 12),
             arrowprops=dict(arrowstyle='->', color='#2E7D32'), fontsize=10, color='#2E7D32')
ax2.set_xlabel('Principale component $k$')
ax2.set_ylabel('Verklaarde variantie (%)')
ax2.set_title('Explained Variance Ratio')
ax2.set_xticks(x_pos)
ax2.legend(loc='center right', framealpha=0.9)
ax2.set_ylim(0, 105)

plt.tight_layout()
plt.show()
