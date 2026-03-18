import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

n = 400
x = np.random.normal(0, 3.0, n)
y = 1.1 * x + np.random.normal(0, 0.9, n)

X = np.column_stack([x, y])
Xc = X - X.mean(axis=0)

cov = (Xc.T @ Xc) / len(Xc)
eigvals, eigvecs = np.linalg.eigh(cov)
idx = np.argsort(eigvals)[::-1]
eigvals = eigvals[idx]
eigvecs = eigvecs[:, idx]

v1 = eigvecs[:, 0]   
v2 = eigvecs[:, 1]   

proj_good = Xc @ v1
proj_bad  = Xc @ v2


fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.3))


color_data = '#2c3e50'
color_pc1 = '#e74c3c'    
color_pc2 = '#3498db'    


ax = axes[0]
ax.scatter(X[:, 0], X[:, 1], c=color_data, alpha=0.75, s=22, edgecolors='none')

scale1 = 8.5
scale2 = 3.2

ax.arrow(0, 0, scale1*v1[0], scale1*v1[1], head_width=0.65, color=color_pc1, 
         lw=3, length_includes_head=True)
ax.arrow(0, 0, scale2*v2[0], scale2*v2[1], head_width=0.55, color=color_pc2, 
         lw=3, length_includes_head=True)

ax.text(scale1*v1[0]*0.95, scale1*v1[1]*0.95, 'PC1', fontsize=12, color=color_pc1, 
        fontweight='bold', ha='center', va='center')
ax.text(scale2*v2[0]*1.1, scale2*v2[1]*1.1 + 0.3, 'PC2', fontsize=12, color=color_pc2, 
        fontweight='bold', ha='center', va='center')

ax.set_xlabel('Originele $x_1$', fontsize=12)
ax.set_ylabel('Originele $x_2$', fontsize=12)
ax.set_title('2D Data met eigenvectoren van $\\Sigma$', fontsize=13, fontweight='bold', pad=15)
ax.grid(True, alpha=0.25)
ax.set_xlim(-10.5, 10.5)
ax.set_ylim(-10, 10)


ax = axes[1]
ax.scatter(proj_good, np.zeros_like(proj_good), c=color_pc1, alpha=0.75, s=20)

ax.set_xlabel('Projectie $x \\cdot v_1$', fontsize=12)
ax.set_title('Goede reductie (op $v_1$)\n$\\lambda_1 = 9.2$ (Grote variantie)', 
             fontsize=12.5, fontweight='bold', pad=10, color=color_pc1)
ax.set_yticks([])
ax.set_xlim(-9, 9)
ax.grid(True, alpha=0.25, axis='x')


ax = axes[2]
ax.scatter(proj_bad, np.zeros_like(proj_bad), c=color_pc2, alpha=0.75, s=20)

ax.set_xlabel('Projectie $x \\cdot v_2$', fontsize=12)
ax.set_title('Slechte reductie (op $v_2$)\n$\\lambda_2 = 0.2$ (Weinig variantie)', 
             fontsize=12.5, fontweight='bold', pad=10, color=color_pc2)
ax.set_yticks([])
ax.set_xlim(-9, 9)
ax.grid(True, alpha=0.25, axis='x')

plt.suptitle('PCA reductie: Goede vs Slechte projectierichting', 
             fontsize=15, fontweight='bold', y=1)

plt.tight_layout()
plt.show()