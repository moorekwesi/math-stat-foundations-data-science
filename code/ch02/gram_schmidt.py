# DEP 803 -- Chapter 2: Linear Algebra for Modelling
# Classical vs modified Gram-Schmidt vs Householder QR: loss of orthogonality
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt


def cgs(A):
    """Classical Gram-Schmidt: project the ORIGINAL column on earlier q's."""
    m, n = A.shape
    Q, R = np.zeros((m, n)), np.zeros((n, n))
    for j in range(n):
        R[:j, j] = Q[:, :j].T @ A[:, j]
        v = A[:, j] - Q[:, :j] @ R[:j, j]
        R[j, j] = np.linalg.norm(v)
        Q[:, j] = v / R[j, j]
    return Q, R


def mgs(A):
    """Modified Gram-Schmidt: remove each q from the UPDATED remaining columns."""
    V = A.astype(float).copy()
    m, n = A.shape
    Q, R = np.zeros((m, n)), np.zeros((n, n))
    for j in range(n):
        R[j, j] = np.linalg.norm(V[:, j])
        Q[:, j] = V[:, j] / R[j, j]
        R[j, j + 1:] = Q[:, j] @ V[:, j + 1:]
        V[:, j + 1:] -= np.outer(Q[:, j], R[j, j + 1:])
    return Q, R


def test_matrix(m, n, kappa, rng):
    """Random m x n matrix with prescribed 2-norm condition number."""
    U, _ = np.linalg.qr(rng.normal(size=(m, n)))
    V, _ = np.linalg.qr(rng.normal(size=(n, n)))
    return U @ np.diag(np.logspace(0, -np.log10(kappa), n)) @ V.T


rng = np.random.default_rng(803)
kappas = np.logspace(1, 15, 15)
loss = {"Classical GS": [], "Modified GS": [], "Householder (np.linalg.qr)": []}
print(" cond(A)     CGS        MGS     Householder   ||A - QR||/||A|| (MGS)")
for i, kap in enumerate(kappas):
    A = test_matrix(80, 12, kap, rng)
    for name, f in zip(loss, [cgs, mgs, np.linalg.qr]):
        Q, R = f(A)
        loss[name].append(np.linalg.norm(Q.T @ Q - np.eye(12), 2))
    Qm, Rm = mgs(A)
    if i % 3 == 0:
        res = np.linalg.norm(A - Qm @ Rm) / np.linalg.norm(A)
        print(f"{kap:8.0e}  " + "  ".join(f"{loss[k][-1]:9.1e}" for k in loss)
              + f"     {res:8.1e}")

plt.figure(figsize=(6, 3.8))
for (name, vals), mk in zip(loss.items(), ["o", "s", "^"]):
    plt.loglog(kappas, vals, marker=mk, ms=4, label=name)
plt.loglog(kappas, 1e-16 * kappas, "k:", lw=1, label=r"$\epsilon\,\kappa$")
plt.loglog(kappas, 1e-16 * kappas ** 2, "k--", lw=1, label=r"$\epsilon\,\kappa^2$")
plt.ylim(1e-17, 10)
plt.xlabel(r"condition number $\kappa(A)$")
plt.ylabel(r"$\|Q^\top Q - I\|_2$")
plt.legend(fontsize=8)
plt.grid(alpha=0.3, which="major")
plt.tight_layout()
plt.savefig("ch02_gram_schmidt.pdf", bbox_inches="tight")
