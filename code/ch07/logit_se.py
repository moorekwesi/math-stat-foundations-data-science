# DEP 803 -- Chapter 7: Estimation Principles
# Standard errors of a logit MLE: analytic Hessian, BFGS approximation, simulation.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import optimize, special

rng = np.random.default_rng(803)
n, R = 500, 2000
b_true = np.array([0.5, -0.6, -0.4])
names = ["const", "income", "distance"]


def make_x():
    return np.column_stack([np.ones(n), rng.normal(size=n),
                            rng.uniform(0, 5, size=n)])


def newton_logit(X, y, tol=1e-10):
    """Newton-Raphson (= Fisher scoring for the canonical logit link)."""
    b = np.zeros(X.shape[1])
    for _ in range(50):
        p = special.expit(X @ b)
        H = (X * (p * (1 - p))[:, None]).T @ X      # information matrix
        step = np.linalg.solve(H, X.T @ (y - p))
        b += step
        if np.max(np.abs(step)) < tol:
            break
    p = special.expit(X @ b)
    H = (X * (p * (1 - p))[:, None]).T @ X
    return b, H


# One data set: compare analytic and BFGS-approximate standard errors
X = make_x()
y = rng.binomial(1, special.expit(X @ b_true))
b, H = newton_logit(X, y)
se_h = np.sqrt(np.diag(np.linalg.inv(H)))
nll = lambda c: np.sum(np.logaddexp(0, X @ c) - y * (X @ c))
gr = lambda c: X.T @ (special.expit(X @ c) - y)
rb = optimize.minimize(nll, np.zeros(3), jac=gr, method="BFGS")
se_bfgs = np.sqrt(np.diag(rb.hess_inv))

# Monte Carlo: sampling sd and coverage of Wald intervals
B = np.empty((R, 3))
cover = np.zeros(3)
for r in range(R):
    Xr = make_x()
    yr = rng.binomial(1, special.expit(Xr @ b_true))
    br, Hr = newton_logit(Xr, yr)
    B[r] = br
    cover += np.abs(br - b_true) < 1.96 * np.sqrt(np.diag(np.linalg.inv(Hr)))
print(f"Logit take-up model, n = {n}; Monte Carlo R = {R}")
print(f"{'':>9} {'b_hat':>8} {'se Hess':>8} {'se BFGS':>8} {'MC sd':>8}"
      f" {'cover':>7}")
for j, nm in enumerate(names):
    print(f"{nm:>9} {b[j]:8.4f} {se_h[j]:8.4f} {se_bfgs[j]:8.4f}"
          f" {B[:, j].std():8.4f} {cover[j] / R:7.3f}")
