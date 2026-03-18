
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker


rng = np.random.default_rng(42)

true_eigenvalues = np.array([8.0, 4.5, 2.8, 0.6, 0.4, 0.3, 0.2, 0.15, 0.1, 0.05])
d = len(true_eigenvalues)
n = 200


Q, _ = np.linalg.qr(rng.standard_normal((d, d)))
Sigma_true = Q @ np.diag(true_eigenvalues) @ Q.T

X = rng.multivariate_normal(np.zeros(d), Sigma_true, size=n)
X -= X.mean(axis=0)

Sigma_hat = (X.T @ X) / n
eigenvalues = np.sort(np.linalg.eigvalsh(Sigma_hat))[::-1]

evr = np.cumsum(eigenvalues) / eigenvalues.sum()

# ---------- Plot ----------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), dpi=150)

components = np.arange(1, d + 1)

ax1.plot(components, eigenvalues, 'o-', color='#2c5f8a', linewidth=1.8,
         markersize=6, markerfacecolor='#2c5f8a', markeredgecolor='white',
         markeredgewidth=1.2, zorder=3)


elbow_k = 3
ax1.axvline(x=elbow_k, color='#c0392b', linestyle='--', linewidth=1.2, alpha=0.7)
ax1.annotate('elleboog ($k=3$)',
             xy=(elbow_k, eigenvalues[elbow_k - 1]),
             xytext=(elbow_k + 1.5, eigenvalues[elbow_k - 1] + 1.8),
             fontsize=10,
             arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.3),
             color='#c0392b')

ax1.set_xlabel('Principale component $k$', fontsize=11)
ax1.set_ylabel('Eigenwaarde $\\lambda_k$', fontsize=11)
ax1.set_title('Scree plot', fontsize=12, fontweight='bold')
ax1.set_xticks(components)
ax1.set_xlim(0.5, d + 0.5)
ax1.set_ylim(bottom=0)
ax1.grid(True, alpha=0.3, linewidth=0.5)

ax2.plot(components, evr * 100, 's-', color='#2c5f8a', linewidth=1.8,
         markersize=6, markerfacecolor='#2c5f8a', markeredgecolor='white',
         markeredgewidth=1.2, zorder=3)

ax2.axhline(y=95, color='#7f8c8d', linestyle=':', linewidth=1.2, alpha=0.8)
ax2.text(d - 0.3, 96, '95\\%', fontsize=9, color='#7f8c8d', ha='right')

evr_at_elbow = evr[elbow_k - 1] * 100
ax2.plot(elbow_k, evr_at_elbow, 'o', color='#c0392b', markersize=9, zorder=4)
ax2.annotate(f'$k=3$: EVR $\\approx$ {evr_at_elbow:.1f}\\%',
             xy=(elbow_k, evr_at_elbow),
             xytext=(elbow_k + 2, evr_at_elbow - 12),
             fontsize=10,
             arrowprops=dict(arrowstyle='->', color='#c0392b', lw=1.3),
             color='#c0392b')

ax2.set_xlabel('Aantal componenten $k$', fontsize=11)
ax2.set_ylabel('Cumulatieve EVR (\\%)', fontsize=11)
ax2.set_title('Explained Variance Ratio', fontsize=12, fontweight='bold')
ax2.set_xticks(components)
ax2.set_xlim(0.5, d + 0.5)
ax2.set_ylim(0, 105)
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
ax2.grid(True, alpha=0.3, linewidth=0.5)

plt.tight_layout()
plt.savefig('elbow_plot.pdf', bbox_inches='tight')
plt.savefig('elbow_plot.png', bbox_inches='tight', dpi=200)  
print("Opgeslagen: elbow_plot.pdf en elbow_plot.png")
plt.show()
