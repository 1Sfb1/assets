import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
n = 150

# Data genereren
cell_size_benign = np.random.normal(50, 12, n)
biomarker_benign = np.random.normal(2.0, 0.4, n)
cell_size_malignant = np.random.normal(50, 12, n)
biomarker_malignant = np.random.normal(3.5, 0.4, n)

X = np.vstack([
    np.column_stack([cell_size_benign, biomarker_benign]),
    np.column_stack([cell_size_malignant, biomarker_malignant])
])

# Centreer de data
Xc = X - X.mean(axis=0)

# Covariantiematrix en eigenwaarden
Sigma = (Xc.T @ Xc) / len(Xc)
eigenvalues = np.linalg.eigvalsh(Sigma)
eigenvalues = eigenvalues[::-1]          # Sorteer van groot naar klein
evr = eigenvalues / eigenvalues.sum() * 100

# Kleuren
c_pc1 = '#e67e22'
c_pc2 = '#2ecc71'

# Plot
plt.figure(figsize=(8, 6))

bars = plt.bar(['PC1', 'PC2'], evr, 
               color=[c_pc1, c_pc2], 
               edgecolor='black', 
               linewidth=0.7)

plt.ylabel('Verklaarde variantie (%)', fontsize=12)
plt.title('Explained Variance Ratio (EVR)', fontsize=14, fontweight='bold', pad=20)
plt.ylim(0, 105)
plt.grid(True, alpha=0.2, axis='y')

# Waarden boven de bars
for bar, value in zip(bars, evr):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
             f'{value:.1f}%', 
             ha='center', 
             fontsize=13, 
             fontweight='bold')

# Waarschuwingstekst
plt.text(0.5, 72, 'EVR ≈ 99% \n maar de discriminiatieve \n informatie zit in PC2!',
         ha='center', va='center',
         fontsize=11, 
         bbox=dict(boxstyle='round,pad=0.8', facecolor='#fff3cd', 
                   edgecolor='#ffc107', linewidth=1.5))

plt.tight_layout()
plt.show()

print("EVR Plot gegenereerd")
print(f"PC1 verklaart: {evr[0]:.2f}%")
print(f"PC2 verklaart: {evr[1]:.2f}%")