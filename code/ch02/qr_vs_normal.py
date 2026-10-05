# DEP 803 -- Chapter 2: Linear Algebra for Modelling
# Householder QR vs normal equations for ill-conditioned polynomial regression
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg


def householder_qr(A):
    """Return R and the Householder vectors (Q kept in factored form)."""
    R = A.astype(float).copy()
    m, n = R.shape
    vs = []
    for j in range(n):
        x = R[j:, j]
        v = x.copy()
        v[0] += np.copysign(np.linalg.norm(x), x[0])   # avoid cancellation
        v /= np.linalg.norm(v)
        R[j:, j:] -= 2.0 * np.outer(v, v @ R[j:, j:])  # apply H = I - 2vv^T
        vs.append(v)
    return np.triu(R[:n]), vs


def apply_qt(vs, b):
    """Compute Q^T b using the stored Householder vectors."""
    b = b.astype(float).copy()
    for j, v in enumerate(vs):
        b[j:] -= 2.0 * v * (v @ b[j:])
    return b


def ls_householder(X, y):
    R, vs = householder_qr(X)
    return linalg.solve_triangular(R, apply_qt(vs, y)[:X.shape[1]])


def ls_normal(X, y):
    return linalg.cho_solve(linalg.cho_factor(X.T @ X), X.T @ y)


rng = np.random.default_rng(803)
n = 100
t = np.linspace(0, 1, n)
degrees = np.arange(1, 14)
errs = {"normal equations (Cholesky)": [], "Householder QR (own code)": [],
        "np.linalg.lstsq (SVD)": []}
conds = []
print("deg   cond(X)    normal eq.  Householder    lstsq")
for d in degrees:
    X = np.vander(t, d + 1, increasing=True)          # 1, t, t^2, ..., t^d
    beta = rng.normal(size=d + 1)
    y = X @ beta                                       # exact fit: zero residual
    conds.append(np.linalg.cond(X))
    sols = [ls_normal(X, y), ls_householder(X, y),
            np.linalg.lstsq(X, y, rcond=None)[0]]
    for key, b in zip(errs, sols):
        errs[key].append(np.linalg.norm(b - beta) / np.linalg.norm(beta))
    if d % 2 == 1:
        print(f"{d:3d}  {conds[-1]:9.2e}  " +
              "  ".join(f"{errs[k][-1]:10.2e}" for k in errs))

conds = np.array(conds)
plt.figure(figsize=(6, 3.8))
for (key, v), mk in zip(errs.items(), ["o", "s", "^"]):
    plt.semilogy(degrees, np.maximum(v, 1e-17), marker=mk, ms=4, label=key)
plt.semilogy(degrees, 1.1e-16 * conds, "k:", lw=1, label=r"$\epsilon\,\kappa(X)$")
plt.semilogy(degrees, np.minimum(1.1e-16 * conds ** 2, 1e3), "k--", lw=1,
             label=r"$\epsilon\,\kappa(X)^2$")
plt.xlabel("polynomial degree $d$")
plt.ylabel("relative coefficient error")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ch02_qr_vs_normal.pdf", bbox_inches="tight")
