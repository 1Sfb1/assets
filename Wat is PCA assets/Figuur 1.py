import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(0)
n = 300


def cov_1_over_n(X: np.ndarray) -> np.ndarray:
    Xc = X - X.mean(axis=0, keepdims=True)
    return (Xc.T @ Xc) / Xc.shape[0]


def rho_from_sigma(S: np.ndarray) -> float:
    return S[0, 1] / np.sqrt(S[0, 0] * S[1, 1])


x = rng.normal(0, 1, n)
eps = rng.normal(0, 1, n)

datasets = [
    ("Sterk positief", np.c_[x, 0.9 * x + 0.3 * eps]),
    ("Sterk negatief", np.c_[x, -0.9 * x + 0.3 * eps]),
    ("Rond 0", np.c_[rng.normal(0, 1, n), rng.normal(0, 1, n)]),
]

fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharex=False, sharey=False)

for ax, (name, X) in zip(axes, datasets):
    S = cov_1_over_n(X)
    rho = rho_from_sigma(S)

    ax.scatter(X[:, 0], X[:, 1], s=10, alpha=0.5)
    ax.set_title(f"{name}\nΣ12={S[0,1]:.2f}, ρ={rho:.2f}")
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")

    sigma_text = np.array2string(S, precision=2, suppress_small=True)
    ax.text(
        0.02,
        0.98,
        f"Σ =\n{sigma_text}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        family="monospace",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
    )

plt.tight_layout()
plt.show()