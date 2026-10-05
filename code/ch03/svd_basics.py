# DEP 803 -- Chapter 3: Eigenvalues, the SVD, PCA and Conditioning
# The SVD, its link to eig(X'X) and eig(XX'), and the Eckart-Young theorem.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
np.set_printoptions(precision=4, suppress=True)
n, p = 8, 4
X = rng.standard_normal((n, p)) @ np.diag([3.0, 2.0, 1.0, 0.2])

U, s, Vt = np.linalg.svd(X, full_matrices=False)    # thin SVD
print("shapes U, s, Vt     :", U.shape, s.shape, Vt.shape)
print("singular values     :", s)
print("||X - U S V'||      : %.2e" % np.linalg.norm(X - U @ np.diag(s) @ Vt))
print("U'U = I, V'V = I    :", np.allclose(U.T @ U, np.eye(p)),
      np.allclose(Vt @ Vt.T, np.eye(p)))

# Relation to eigen-decompositions
ev_XtX = np.linalg.eigvalsh(X.T @ X)[::-1]
ev_XXt = np.linalg.eigvalsh(X @ X.T)[::-1]
print("sqrt eig(X'X)       :", np.sqrt(ev_XtX))
print("eig(XX') (top 5)    :", ev_XXt[:5])
print("s^2                 :", s ** 2)

# Rank and condition number from the singular values
print("rank, cond(X)       :", np.linalg.matrix_rank(X), round(s[0] / s[-1], 3),
      round(np.linalg.cond(X), 3))

# Eckart-Young-Mirsky: best rank-k approximation error
print(" k  ||X-Xk||_2  s_(k+1)   ||X-Xk||_F  sqrt(sum s_j^2, j>k)")
for k in range(1, p):
    Xk = U[:, :k] @ np.diag(s[:k]) @ Vt[:k]
    e2 = np.linalg.norm(X - Xk, 2)
    eF = np.linalg.norm(X - Xk, "fro")
    print(f"{k:2d}  {e2:9.4f}  {s[k]:8.4f}   {eF:9.4f}   "
          f"{np.sqrt((s[k:] ** 2).sum()):9.4f}")

# A random rank-2 competitor never does better
best = np.linalg.norm(X - U[:, :2] @ np.diag(s[:2]) @ Vt[:2], "fro")
errs = []
for _ in range(2000):
    A = rng.standard_normal((n, 2)) @ rng.standard_normal((2, p))
    # optimal scaling of the competitor's column space (least squares)
    Qa, _ = np.linalg.qr(A)
    errs.append(np.linalg.norm(X - Qa @ (Qa.T @ X), "fro"))
print("best rank-2 error %.4f; best of 2000 random subspaces %.4f"
      % (best, min(errs)))
