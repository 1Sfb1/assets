import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 150

cell_size_benign = np.random.normal(50, 12, n)
biomarker_benign = np.random.normal(2.0, 0.4, n)
cell_size_malignant = np.random.normal(50, 12, n)
biomarker_malignant = np.random.normal(3.5, 0.4, n)

X = np.vstack([
    np.column_stack([cell_size_benign, biomarker_benign]),
    np.column_stack([cell_size_malignant, biomarker_malignant])
])

Xc = X - X.mean(axis=0)
Sigma = (Xc.T @ Xc) / len(Xc)

plt.figure(figsize=(7, 6))
im = plt.imshow(Sigma, cmap='RdBu_r', vmin=-5, vmax=150)

for i in range(2):
    for j in range(2):
        color = 'white' if abs(Sigma[i,j]) > 50 else 'black'
        plt.text(j, i, f'{Sigma[i,j]:.2f}', ha='center', va='center',
                 fontsize=15, fontweight='bold', color=color)

plt.xticks([0, 1], ['Celgrootte', 'Biomarker'], fontsize=11)
plt.yticks([0, 1], ['Celgrootte', 'Biomarker'], fontsize=11)
plt.title('Covariantiematrix $\\Sigma$', fontsize=14, fontweight='bold', pad=15)
plt.colorbar(im, fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()