# DEP 803 -- Chapter 3: Eigenvalues, the SVD, PCA and Conditioning
# PageRank of a simulated inter-district trade network; eig versus eigh.
# (c) 2026 Stephen E. Moore. MIT Licence.
import time
import numpy as np

rng = np.random.default_rng(803)
n = 10
names = [f"D{i:02d}" for i in range(n)]
# Simulated directed trade links: W[i, j] = value shipped from i to j.
# District D00 is a hub port that many districts ship to.
W = rng.gamma(0.6, 1.0, size=(n, n)) * (rng.random((n, n)) < 0.4)
W[:, 0] += rng.uniform(1.0, 3.0, n)
np.fill_diagonal(W, 0.0)
W[7, :] = 0.0                          # D07 is a "dangling" node (no exports)

# Column-stochastic link matrix: M[j, i] = share of i's flows going to j
out = W.sum(axis=1)
M = np.where(out[:, None] > 0, W / np.where(out > 0, out, 1)[:, None], 1 / n).T
alpha = 0.85
G = alpha * M + (1 - alpha) / n * np.ones((n, n))   # the Google matrix

# Power iteration
r = np.full(n, 1 / n)
for k in range(1, 200):
    r_new = G @ r
    if np.abs(r_new - r).sum() < 1e-12:
        break
    r = r_new
print(f"power iteration converged in {k} steps")

# Dense eigen-solver for comparison
lam, V = np.linalg.eig(G)
i1 = np.argmax(np.abs(lam))
v = np.real(V[:, i1])
v = v / v.sum()
lam2 = np.sort(np.abs(lam))[-2]
print(f"lambda_1 = {np.real(lam[i1]):.6f}, |lambda_2| = {lam2:.4f}"
      f" (bound alpha = {alpha})")
print("max |r_power - r_eig| = %.2e" % np.abs(r_new - v).max())
rank = np.argsort(-v)
print("rank district PageRank  in-flow share")
inflow = W.sum(axis=0) / W.sum()
for pos, i in enumerate(rank[:5], start=1):
    print(f"{pos:4d}   {names[i]}    {v[i]:.4f}     {inflow[i]:.4f}")

# eig versus eigh on a symmetric 600 x 600 matrix
B = rng.standard_normal((600, 600))
S = (B + B.T) / 2
t0 = time.perf_counter(); l_g, V_g = np.linalg.eig(S); t1 = time.perf_counter()
l_h, V_h = np.linalg.eigh(S); t2 = time.perf_counter()
print(f"eig : {t1 - t0:.3f}s, dtype {l_g.dtype}, "
      f"orthogonality err {np.abs(V_g.T @ V_g - np.eye(600)).max():.1e}")
print(f"eigh: {t2 - t1:.3f}s, dtype {l_h.dtype}, "
      f"orthogonality err {np.abs(V_h.T @ V_h - np.eye(600)).max():.1e}")
print("same spectrum:", np.allclose(np.sort(l_g.real), l_h))
