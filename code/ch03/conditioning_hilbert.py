# DEP 803 -- Chapter 3: Eigenvalues, the SVD, PCA and Conditioning
# Condition numbers, the perturbation bound, Hilbert matrices, LS conditioning.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import hilbert

rng = np.random.default_rng(803)
eps = np.finfo(float).eps

# 1. The perturbation bound ||dx||/||x|| <= kappa ||db||/||b|| is attainable
A = np.array([[1.0, 1.0], [1.0, 1.0001]])
U, s, Vt = np.linalg.svd(A)
kappa = s[0] / s[-1]
b = U[:, 0]                       # b along the top left singular vector
db = 1e-6 * U[:, -1]              # perturbation along the bottom one
x = np.linalg.solve(A, b)
dx = np.linalg.solve(A, b + db) - x
amp = (np.linalg.norm(dx) / np.linalg.norm(x)) / (np.linalg.norm(db)
                                                   / np.linalg.norm(b))
print(f"kappa_2(A) = {kappa:.1f}; observed amplification = {amp:.1f}")

# 2. Hilbert matrices H_ij = 1/(i+j-1): solve H x = H 1
print(" n   cond_2(H)   rel. error   kappa*eps")
ns = np.arange(2, 15)
conds, errs = [], []
for n in ns:
    H = hilbert(n)
    x_true = np.ones(n)
    x_hat = np.linalg.solve(H, H @ x_true)
    c = np.linalg.cond(H)
    e = np.linalg.norm(x_hat - x_true) / np.linalg.norm(x_true)
    conds.append(c); errs.append(e)
    if n % 2 == 0:
        print(f"{n:2d}  {c:10.2e}  {e:10.2e}  {c * eps:10.2e}")

# 3. Least squares: normal equations square the condition number
t = np.linspace(0, 1, 100)
X = np.vander(t, 9, increasing=True)          # polynomial regression, degree 8
beta = rng.standard_normal(9)
y = X @ beta
b_ne = np.linalg.solve(X.T @ X, X.T @ y)
Q, R = np.linalg.qr(X)
b_qr = np.linalg.solve(R, Q.T @ y)
print(f"cond(X) = {np.linalg.cond(X):.2e}, "
      f"cond(X'X) = {np.linalg.cond(X.T @ X):.2e}")
for name, bh in [("normal equations", b_ne), ("QR", b_qr)]:
    print(f"{name:17s}: rel. coef. error "
          f"{np.linalg.norm(bh - beta) / np.linalg.norm(beta):.2e}")

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.semilogy(ns, conds, "o-", label="cond_2(H_n)")
ax.semilogy(ns, errs, "s-", label="relative error of solve")
ax.semilogy(ns, np.array(conds) * eps, "k:", label="cond * eps_mach")
ax.axhline(1.0, color="gray", lw=0.8)
ax.set_xlabel("dimension n")
ax.set_ylabel("log scale")
ax.set_title("Hilbert matrices: error tracks cond(H) * eps_mach")
ax.legend(frameon=False)
plt.savefig("ch03_hilbert.pdf", bbox_inches="tight")
