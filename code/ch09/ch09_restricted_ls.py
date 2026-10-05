# DEP 803 -- Chapter 9: Constrained Optimisation and Lagrange Multipliers
# Restricted least squares: closed form, F test, check with SLSQP.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import stats
from scipy.optimize import minimize

rng = np.random.default_rng(803)
n = 200                                    # simulated farm survey
logK = rng.normal(1.0, 0.6, n)             # log capital
logL = 0.5 * logK + rng.normal(0.8, 0.5, n)  # log labour (correlated)
logA = rng.normal(0.0, 0.4, n)             # log land
X = np.column_stack([np.ones(n), logK, logL, logA])
beta_true = np.array([0.5, 0.35, 0.45, 0.20])  # constant returns: sum = 1
y = X @ beta_true + rng.normal(0, 0.3, n)

XtX_inv = np.linalg.inv(X.T @ X)
b = XtX_inv @ X.T @ y                       # unrestricted OLS


def restricted(R, r):
    """b_R = b - (X'X)^-1 R' [R (X'X)^-1 R']^-1 (R b - r)."""
    W = np.linalg.inv(R @ XtX_inv @ R.T)
    lam = W @ (R @ b - r)                   # (scaled) Lagrange multipliers
    return b - XtX_inv @ R.T @ lam, lam


def f_test(R, r):
    bR, _ = restricted(R, r)
    ssr_u = np.sum((y - X @ b)**2)
    ssr_r = np.sum((y - X @ bR)**2)
    q, k = R.shape[0], X.shape[1]
    F = ((ssr_r - ssr_u) / q) / (ssr_u / (n - k))
    return F, stats.f.sf(F, q, n - k)


print("OLS (unrestricted):       ", np.round(b, 4), " sum of slopes = %.4f"
      % b[1:].sum())
R1, r1 = np.array([[0.0, 1, 1, 1]]), np.array([1.0])   # constant returns
bR, lam = restricted(R1, r1)
print("Restricted (CRS):         ", np.round(bR, 4), " sum of slopes = %.4f"
      % bR[1:].sum())

# numerical check: minimise SSR subject to R b = r with SLSQP
obj = lambda be: np.sum((y - X @ be)**2)
out = minimize(obj, x0=np.zeros(4), method="SLSQP", tol=1e-12,
               constraints=[{"type": "eq", "fun": lambda be: R1 @ be - r1}])
print("SLSQP check:              ", np.round(out.x, 4))
print("multiplier of the SSR problem (2*lam): %.4f" % (2 * lam[0]))
F, pv = f_test(R1, r1)
print("\nH0 constant returns (1 restriction):  F = %.3f   p = %.3f" % (F, pv))

R2 = np.array([[0.0, 1, 1, 1], [0.0, 1, -1, 0]])       # CRS and bK = bL
r2 = np.array([1.0, 0.0])
F2, pv2 = f_test(R2, r2)
print("H0 CRS and bK = bL (2 restrictions): F = %.3f   p = %.4f" % (F2, pv2))
# variance reduction from imposing a true restriction
s2 = np.sum((y - X @ b)**2) / (n - 4)
V_u = s2 * XtX_inv
M = XtX_inv @ R1.T @ np.linalg.inv(R1 @ XtX_inv @ R1.T) @ R1
V_r = (np.eye(4) - M) @ V_u
print("\nstd errors unrestricted:", np.round(np.sqrt(np.diag(V_u)), 4))
print("std errors restricted:  ", np.round(np.sqrt(np.diag(V_r)), 4))
