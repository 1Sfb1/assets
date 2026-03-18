import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
import matplotlib.colors as mcolors

data = load_breast_cancer()
X = StandardScaler().fit_transform(data.data)
feature_names = data.feature_names


corr = np.corrcoef(X.T)

short_names = []
for name in feature_names:
    name = name.replace('mean ', 'μ ')
    name = name.replace('worst ', 'w ')
    name = name.replace('error ', 'σ ')
    name = name.replace('se ', 'σ ')
    name = name.replace('radius', 'rad')
    name = name.replace('texture', 'tex')
    name = name.replace('perimeter', 'peri')
    name = name.replace('area', 'area')
    name = name.replace('smoothness', 'smooth')
    name = name.replace('compactness', 'compact')
    name = name.replace('concavity', 'concav')
    name = name.replace('concave points', 'conc.pts')
    name = name.replace('symmetry', 'sym')
    name = name.replace('fractal dimension', 'frac.dim')
    name = name.replace('fractal_dimension', 'frac.dim')
    short_names.append(name)


fig, ax = plt.subplots(figsize=(9, 7.5))
cmap = plt.cm.RdBu_r

im = ax.imshow(corr, cmap=cmap, vmin=-1, vmax=1, aspect='equal')

ax.set_xticks(range(30))
ax.set_yticks(range(30))
ax.set_xticklabels(short_names, rotation=65, ha='right', fontsize=7.5)
ax.set_yticklabels(short_names, fontsize=7.5)


for pos in [9.5, 19.5]:
    ax.axhline(y=pos, color='black', linewidth=1.5, alpha=0.7)
    ax.axvline(x=pos, color='black', linewidth=1.5, alpha=0.7)


ax2 = ax.secondary_xaxis('top')
ax2.set_xticks([4.5, 14.5, 24.5])
ax2.set_xticklabels(['Gemiddelde (μ)', 'Standaardfout (σ)', 'Worst (w)'],
                      fontsize=10, fontweight='bold')
ax2.tick_params(length=0, pad=8)

cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
cbar.set_label('Pearson-correlatiecoëfficiënt $\\rho$', fontsize=12)
cbar.ax.tick_params(labelsize=9)

ax.set_title('Correlatiematrix: Breast Cancer Wisconsin ($d = 30$)',
             fontsize=14, fontweight='bold', pad=20)

high_corr = np.sum(np.abs(corr[np.triu_indices(30, k=1)]) > 0.7)
total_pairs = 30 * 29 // 2
info = f'{high_corr} van {total_pairs} paren hebben $|\\rho| > 0.7$'
ax.text(0.5, -0.08, info, transform=ax.transAxes,
        ha='center', fontsize=11, style='italic',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', edgecolor='gray', alpha=0.9))

plt.tight_layout()
plt.show()

print(f"High correlation pairs (|rho| > 0.7): {high_corr} / {total_pairs}")
print(f"High correlation pairs (|rho| > 0.9): {np.sum(np.abs(corr[np.triu_indices(30, k=1)]) > 0.9)} / {total_pairs}")
print("Done!")
