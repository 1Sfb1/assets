import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

np.random.seed(42)

n = 150

cell_size_benign = np.random.normal(50, 12, n)
biomarker_benign = np.random.normal(2.0, 0.4, n)

cell_size_malignant = np.random.normal(50, 12, n)
biomarker_malignant = np.random.normal(3.5, 0.4, n)

X0 = np.column_stack([cell_size_benign, biomarker_benign])
X1 = np.column_stack([cell_size_malignant, biomarker_malignant])
X = np.vstack([X0, X1])
labels = np.array([0] * n + [1] * n)

mean = X.mean(axis=0)
Xc = X - mean

Sigma = (Xc.T @ Xc) / len(Xc)
eigenvalues, eigenvectors = np.linalg.eigh(Sigma)

idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

pc1 = eigenvectors[:, 0]
pc2 = eigenvectors[:, 1]

proj_pc1 = Xc @ pc1
proj_pc2 = Xc @ pc2

fig = plt.figure(figsize=(16, 10), facecolor='white')
gs = gridspec.GridSpec(2, 3, height_ratios=[1.3, 1], hspace=0.35, wspace=0.35)

c_benign = '#3498db'
c_malignant = '#e74c3c'
c_pc1 = '#e67e22'
c_pc2 = '#2ecc71'

ax0 = fig.add_subplot(gs[0, :2])
ax0.scatter(X0[:, 0], X0[:, 1], c=c_benign, alpha=0.5, s=25, label='Benigne', edgecolors='none')
ax0.scatter(X1[:, 0], X1[:, 1], c=c_malignant, alpha=0.5, s=25, label='Maligne', edgecolors='none')

origin = mean
scale1 = 18
scale2 = 5

ax0.annotate('', xy=(origin[0] + scale1*pc1[0], origin[1] + scale1*pc1[1]),
             xytext=(origin[0], origin[1]),
             arrowprops=dict(arrowstyle='->', color=c_pc1, lw=2.5))
ax0.annotate('', xy=(origin[0] + scale2*pc2[0], origin[1] + scale2*pc2[1]),
             xytext=(origin[0], origin[1]),
             arrowprops=dict(arrowstyle='->', color=c_pc2, lw=2.5))

ax0.text(origin[0] + scale1*pc1[0] + 1, origin[1] + scale1*pc1[1] - 0.15,
         f'PC1  ($\\lambda_1 = {eigenvalues[0]:.1f}$)', fontsize=11, color=c_pc1, fontweight='bold')

ax0.set_xlabel('Feature 1: Celgrootte (willekeurige eenheden)', fontsize=11)
ax0.set_ylabel('Feature 2: Biomarkerconcentratie (ng/mL)', fontsize=11)
ax0.set_title('(a) Twee klassen in de originele ruimte $\\mathbb{R}^2$', fontsize=13, fontweight='bold')
ax0.legend(fontsize=10, loc='upper left')


plt.tight_layout()
plt.show()

