# DEP 803 -- Chapter 12: Matrix Calculus, Regularisation and Bias-Variance
# Training vs test error, and effective degrees of freedom tr(H).
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from numpy.polynomial import legendre

rng = np.random.default_rng(12)
f = lambda x: np.sin(1.5 * np.pi * x) + 0.5 * x**2
sigma, n = 0.35, 30
x = rng.uniform(-1, 1, n)
y = f(x) + sigma * rng.normal(size=n)
xt = rng.uniform(-1, 1, 5000)                 # large test sample
yt = f(xt) + sigma * rng.normal(size=xt.size)

degrees = np.arange(0, 16)
tr_err, te_err, edf = [], [], []
for deg in degrees:
    Xd = legendre.legvander(x, deg)
    Hat = Xd @ np.linalg.pinv(Xd)              # hat (projection) matrix
    coef = np.linalg.pinv(Xd) @ y
    tr_err.append(np.mean((y - Hat @ y) ** 2))
    te_err.append(np.mean((yt - legendre.legvander(xt, deg) @ coef) ** 2))
    edf.append(np.trace(Hat))
print(f"{'deg':>3}{'tr(H)':>7}{'train MSE':>11}{'test MSE':>10}")
for k in range(0, 16, 3):
    print(f"{degrees[k]:>3}{edf[k]:>7.2f}{tr_err[k]:>11.4f}{te_err[k]:>10.4f}")

# ridge on a degree-15 basis: edf = sum d_j^2 / (d_j^2 + lambda)
Xd = legendre.legvander(x, 15)[:, 1:]
Xc = Xd - Xd.mean(0)
d = np.linalg.svd(Xc, compute_uv=False)
print("\nridge on degree-15 basis (centred, intercept unpenalised):")
for lam in [1e-6, 1e-3, 1e-1, 10.0]:
    Hl = Xc @ np.linalg.solve(Xc.T @ Xc + lam * np.eye(15), Xc.T)
    print(f"lambda = {lam:<7g} edf = 1 + tr(H) = {1 + np.trace(Hl):6.2f}"
          f"   via SVD = {1 + np.sum(d**2 / (d**2 + lam)):6.2f}")

plt.figure(figsize=(6, 3.8))
plt.plot(edf, tr_err, "o-", label="training MSE")
plt.plot(edf, te_err, "s-", label="test MSE")
plt.axhline(sigma**2, color="gray", ls="--", label=r"noise $\sigma^2$")
plt.yscale("log")
plt.ylim(0.02, 20)
plt.xlabel("effective degrees of freedom tr(H) = degree + 1")
plt.ylabel("mean squared error (log scale)")
plt.title("Underfitting and overfitting (one sample, n = 30)")
plt.legend(fontsize=9)
plt.grid(alpha=0.3)
plt.savefig("ch12_train_test.pdf", bbox_inches="tight")
