# DEP 803 -- Chapter 3: Eigenvalues, the SVD, PCA and Conditioning
# Spectral theorem, definiteness tests and Mahalanobis ellipses for a covariance.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
np.set_printoptions(precision=4, suppress=True)

# Simulated household survey: log income and log consumption (centred)
Sigma_true = np.array([[1.00, 0.72],
                       [0.72, 0.64]])
X = rng.multivariate_normal([0, 0], Sigma_true, size=500)
S = np.cov(X, rowvar=False)
lam, Q = np.linalg.eigh(S)             # ascending eigenvalues, orthonormal Q
print("sample covariance S:\n", S)
print("eigenvalues        :", lam)
print("Q'Q = I ?          :", np.allclose(Q.T @ Q, np.eye(2)))
print("||S - Q L Q'||     : %.2e" % np.linalg.norm(S - Q @ np.diag(lam) @ Q.T))

# Three tests of positive definiteness
minors = [np.linalg.det(S[:k, :k]) for k in (1, 2)]
print("leading minors     :", np.round(minors, 4))
L = np.linalg.cholesky(S)              # succeeds only if S is PD
print("Cholesky factor L  :\n", L)
B = np.array([[1.0, 2.0], [2.0, 1.0]])   # indefinite
print("eig of indefinite B:", np.linalg.eigvalsh(B))
try:
    np.linalg.cholesky(B)
except np.linalg.LinAlgError as err:
    print("Cholesky of B fails:", err)

# Mahalanobis vs Euclidean distance for two candidate points
Sinv = np.linalg.inv(S)
pts = np.array([[1.5, 1.5], [1.5, -0.5]])
for p in pts:
    d_e = np.linalg.norm(p)
    d_m = np.sqrt(p @ Sinv @ p)
    print(f"point {p}: Euclid = {d_e:.3f}, Mahalanobis = {d_m:.3f}")
d2 = np.einsum("ij,jk,ik->i", X, Sinv, X)
print("share with d^2 > 5.99 (chi2_2 95%%): %.3f" % np.mean(d2 > 5.991))

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.scatter(X[:, 0], X[:, 1], s=6, alpha=0.4, color="gray")
t = np.linspace(0, 2 * np.pi, 200)
circle = np.vstack([np.cos(t), np.sin(t)])
for c in (1, 2, 3):
    ell = Q @ np.diag(np.sqrt(lam)) @ circle * c
    ax.plot(ell[0], ell[1], lw=1.5, color="C0")
for j in range(2):
    v = Q[:, j] * 2 * np.sqrt(lam[j])
    ax.annotate("", xy=v, xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color="C3", lw=2))
ax.plot(*pts.T, "k*", ms=12)
ax.set_aspect("equal")
ax.set_xlabel("log income (centred)")
ax.set_ylabel("log consumption (centred)")
ax.set_title("Mahalanobis ellipses x' S^-1 x = c^2, c = 1, 2, 3")
plt.savefig("ch03_mahalanobis.pdf", bbox_inches="tight")
