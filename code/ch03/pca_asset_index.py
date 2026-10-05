# DEP 803 -- Chapter 3: Eigenvalues, the SVD, PCA and Conditioning
# A DHS-style asset index (Filmer-Pritchett) on a simulated household survey.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)
n = 3000
assets = ["electricity", "piped water", "flush toilet", "television",
          "refrigerator", "radio", "mobile phone", "motorbike", "car",
          "finished floor", "bicycle", "earth floor"]
# Simulated: ownership probability = Phi(a + b * wealth + c * rural)
a = np.array([0.3, -0.4, -0.9, -0.2, -1.0, 0.6, 1.0, -0.8, -1.9, 0.2, -0.3, -0.3])
b = np.array([1.1, 1.0, 1.2, 1.2, 1.3, 0.4, 0.8, 0.5, 1.0, 1.0, -0.1, -1.0])
c = np.array([-.6, -.6, -.5, -.3, -.3, 0.1, -.2, 0.3, -.2, -.4, 0.6, 0.5])
wealth = rng.standard_normal(n)                     # latent, unobserved
rural = (rng.random(n) < 0.5).astype(float)
A = (rng.random((n, 12)) < stats.norm.cdf(a + np.outer(wealth, b)
                                          + np.outer(rural, c))).astype(float)

mu, sd = A.mean(0), A.std(0, ddof=1)
Z = (A - mu) / sd
R = np.corrcoef(A, rowvar=False)
lam, V = np.linalg.eigh(R)
lam, V = lam[::-1], V[:, ::-1]
v1 = V[:, 0] * np.sign(V[:, 0].sum())               # fix the sign convention
print("eigenvalues of R (first 4):", np.round(lam[:4], 3),
      " PC1 share: %.3f" % (lam[0] / 12))
print(f"{'asset':15s} own%   PC1 loading  score weight (v/sd)")
for j in np.argsort(-v1):
    print(f"{assets[j]:15s} {100 * mu[j]:5.1f}  {v1[j]:8.3f}    "
          f"{v1[j] / sd[j]:8.3f}")

index = Z @ v1                                      # Filmer-Pritchett index
count = A.sum(1) - 2 * A[:, -1]                     # naive count (earth floor -)
for name, x in [("PCA index", index), ("asset count", count)]:
    rho = stats.spearmanr(x, wealth).correlation
    poor_true = wealth <= np.quantile(wealth, 0.4)
    # exactly 40% flagged; ties broken by survey order (households are iid)
    poor_hat = stats.rankdata(x, method="ordinal") <= 0.4 * n
    hit = np.mean(poor_hat[poor_true])
    print(f"{name:12s}: Spearman with latent wealth {rho:.3f}; "
          f"poorest-40% correctly targeted {hit:.3f}")
q = np.digitize(index, np.quantile(index, [0.2, 0.4, 0.6, 0.8]))
print("rural share by index quintile:",
      np.round([rural[q == k].mean() for k in range(5)], 3))

scores = Z @ V[:, :2]
fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.6))
o = np.argsort(v1)
axes[0].barh(np.array(assets)[o], v1[o], color="C0")
axes[0].set_xlabel("PC1 loading")
axes[0].set_title("Asset index weights")
idx = rng.choice(n, 400, replace=False)
axes[1].scatter(scores[idx, 0], scores[idx, 1], s=5, c=wealth[idx],
                cmap="viridis", alpha=0.6)
for j in range(12):
    axes[1].arrow(0, 0, 4 * V[j, 0], 4 * V[j, 1], color="C3", width=0.01)
    if abs(V[j, 1]) > 0.3 or abs(V[j, 0]) > 0.35:
        axes[1].text(4.3 * V[j, 0], 4.3 * V[j, 1], assets[j], fontsize=7)
axes[1].set_xlabel("PC1 score"); axes[1].set_ylabel("PC2 score")
axes[1].set_title("Biplot (colour = latent wealth)")
plt.tight_layout()
plt.savefig("ch03_asset_index.pdf", bbox_inches="tight")
