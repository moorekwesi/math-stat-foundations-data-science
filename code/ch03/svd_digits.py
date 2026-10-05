# DEP 803 -- Chapter 3: Eigenvalues, the SVD, PCA and Conditioning
# Low-rank image compression of a mosaic built from sklearn's digits data.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

digits = load_digits()
imgs = digits.images[:100]                 # 100 images of 8 x 8 pixels
# Tile them into an 80 x 80 mosaic (10 x 10 grid of digits)
mosaic = imgs.reshape(10, 10, 8, 8).transpose(0, 2, 1, 3).reshape(80, 80)
U, s, Vt = np.linalg.svd(mosaic, full_matrices=False)

m, n = mosaic.shape
print(f"image {m} x {n}, numerical rank {np.linalg.matrix_rank(mosaic)}")
print(" k   storage   ratio  rel.Frob.err  energy kept")
energy = np.cumsum(s ** 2) / np.sum(s ** 2)
ranks = (1, 5, 10, 20, 30)
approx = {}
for k in ranks:
    Ak = (U[:, :k] * s[:k]) @ Vt[:k]
    approx[k] = Ak
    store = k * (m + n + 1)
    rel = np.linalg.norm(mosaic - Ak) / np.linalg.norm(mosaic)
    print(f"{k:3d}  {store:7d}  {store / (m * n):6.3f}   {rel:8.4f}"
          f"     {energy[k - 1]:.4f}")
k90 = np.searchsorted(energy, 0.90) + 1
print(f"rank needed for 90% of squared Frobenius norm: {k90}")

fig, axes = plt.subplots(1, 4, figsize=(8.5, 2.6))
for ax, k in zip(axes, (5, 10, 20)):
    ax.imshow(approx[k], cmap="gray_r")
    ax.set_title(f"rank {k}")
for ax in axes[:3]:
    ax.set_xticks([]); ax.set_yticks([])
axes[3].imshow(mosaic, cmap="gray_r")
axes[3].set_title(f"original (rank {np.linalg.matrix_rank(mosaic)})")
axes[3].set_xticks([]); axes[3].set_yticks([])
plt.savefig("ch03_svd_digits.pdf", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(6, 3.4))
ax.semilogy(np.arange(1, len(s) + 1), s, "o-", ms=3)
ax.set_xlabel("index j")
ax.set_ylabel("singular value s_j (log scale)")
ax.set_title("Singular value spectrum of the digit mosaic")
plt.savefig("ch03_svd_spectrum.pdf", bbox_inches="tight")
