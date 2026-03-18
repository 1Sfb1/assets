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
_, eigenvectors = np.linalg.eigh((Xc.T @ Xc) / len(Xc))
eigenvectors = eigenvectors[:, np.argsort(np.linalg.eigvalsh((Xc.T @ Xc) / len(Xc)))[::-1]]

proj_pc1 = Xc @ eigenvectors[:, 0]
proj_pc2 = Xc @ eigenvectors[:, 1]
labels = np.array([0]*n + [1]*n)

c_benign = '#3498db'
c_malignant = '#e74c3c'

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# PC1
bins = np.linspace(proj_pc1.min()-2, proj_pc1.max()+2, 40)
ax1.hist(proj_pc1[labels==0], bins=bins, alpha=0.65, color=c_benign, label='Benigne', density=True)
ax1.hist(proj_pc1[labels==1], bins=bins, alpha=0.65, color=c_malignant, label='Maligne', density=True)
ax1.set_xlabel('Projectie op PC1', fontsize=12)
ax1.set_ylabel('Dichtheid', fontsize=12)
ax1.set_title('PCA kiest PC1: klassen overlappen', fontsize=13, fontweight='bold', color='#e67e22')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.2)

# PC2
bins2 = np.linspace(proj_pc2.min()-0.5, proj_pc2.max()+0.5, 40)
ax2.hist(proj_pc2[labels==0], bins=bins2, alpha=0.65, color=c_benign, label='Benigne', density=True)
ax2.hist(proj_pc2[labels==1], bins=bins2, alpha=0.65, color=c_malignant, label='Maligne', density=True)
ax2.set_xlabel('Projectie op PC2', fontsize=12)
ax2.set_ylabel('Dichtheid', fontsize=12)
ax2.set_title('PC2 (verwijderd door PCA): klassen scheiden goed', fontsize=13, fontweight='bold', color='#2ecc71')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.2)

plt.suptitle('Vergelijking van projecties', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.show()