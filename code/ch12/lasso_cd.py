# DEP 803 -- Chapter 12: Matrix Calculus, Regularisation and Bias-Variance
# Lasso by cyclic coordinate descent with soft-thresholding; KKT check.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from sklearn.linear_model import Lasso

rng = np.random.default_rng(803)


def soft(z, t):
    return np.sign(z) * np.maximum(np.abs(z) - t, 0.0)


def lasso_cd(X, y, alpha, b=None, tol=1e-10, max_sweeps=5000):
    """Minimise (1/2n)||y - Xb||^2 + alpha ||b||_1 (sklearn's scaling)."""
    n, p = X.shape
    b = np.zeros(p) if b is None else b.copy()
    col_ss = np.sum(X**2, axis=0) / n
    r = y - X @ b                                # current residual
    for sweep in range(max_sweeps):
        max_step = 0.0
        for j in range(p):
            old = b[j]
            rho = X[:, j] @ r / n + col_ss[j] * old  # partial-residual corr.
            b[j] = soft(rho, alpha) / col_ss[j]
            if b[j] != old:
                r -= X[:, j] * (b[j] - old)
                max_step = max(max_step, abs(b[j] - old))
        if max_step < tol:
            break
    return b, sweep + 1


# simulated targeting problem: 200 candidate indicators, 5 matter
n, p = 100, 200
X = rng.normal(size=(n, p))
X = (X - X.mean(0)) / X.std(0)
beta = np.zeros(p)
beta[[3, 17, 42, 99, 150]] = [2.0, -1.5, 1.0, 0.8, -0.6]
y = X @ beta + 0.5 * rng.normal(size=n)
y = y - y.mean()

alpha = 0.1
b, sweeps = lasso_cd(X, y, alpha)
skl = Lasso(alpha=alpha, fit_intercept=False, tol=1e-12, max_iter=100000)
skl.fit(X, y)
print(f"coordinate descent converged in {sweeps} sweeps")
print(f"max |ours - sklearn| = {np.abs(b - skl.coef_).max():.1e}")
sel = np.flatnonzero(b)
print(f"{len(sel)} selected; {np.sum(beta[sel] != 0)} of 5 true signals among them")
print(f"{'j':>5}{'b_j':>9}{'true':>7}")
for j in sel:
    print(f"{j:>5}{b[j]:>9.4f}{beta[j]:>7.1f}")

# KKT (subgradient) conditions: x_j'(y - Xb)/n = alpha*sign(b_j) if b_j != 0,
# and |x_j'(y - Xb)/n| <= alpha if b_j = 0
c = X.T @ (y - X @ b) / n
act = b != 0
print(f"active set : max |c_j - alpha sign(b_j)| = "
      f"{np.abs(c[act] - alpha * np.sign(b[act])).max():.1e}")
print(f"inactive   : max |c_j| = {np.abs(c[~act]).max():.4f} <= alpha = {alpha}")
print(f"alpha_max = max|x_j'y|/n = {np.abs(X.T @ y).max() / n:.4f} "
      "(all coefficients zero above this)")
