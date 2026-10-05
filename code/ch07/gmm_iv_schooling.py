# DEP 803 -- Chapter 7: Estimation Principles
# Returns to schooling with endogenous schooling: OLS, 2SLS and two-step GMM.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import stats

rng = np.random.default_rng(803)
n = 3000
# Simulated labour-force survey (NOT real data)
ability = rng.normal(size=n)                    # unobserved by the analyst
dist = rng.uniform(0, 10, size=n)               # km to nearest secondary school
prog = rng.binomial(1, 0.4, size=n)             # exposed to school-building drive
v = rng.normal(size=n)
school = 9 - 0.25 * dist + 1.5 * prog + 1.0 * ability + v
u = rng.normal(size=n) * (0.3 + 0.1 * np.abs(school - 9))   # heteroskedastic
logw = 1.0 + 0.08 * school + 0.25 * ability + u            # true return 0.08

def gmm(y, X, Z, W):
    """Linear GMM: minimise gbar' W gbar with g_i = z_i (y_i - x_i'b)."""
    A = X.T @ Z @ W @ Z.T @ X
    return np.linalg.solve(A, X.T @ Z @ W @ Z.T @ y)

def robust_gmm_cov(y, X, Z, W, b):
    e = y - X @ b
    S = (Z * e[:, None] ** 2).T @ Z / n          # (1/n) sum e_i^2 z_i z_i'
    G = Z.T @ X / n
    M = np.linalg.inv(G.T @ W @ G)
    return M @ G.T @ W @ S @ W @ G @ M / n, S

def j_test(y, X, Z, b, S):
    gbar = Z.T @ (y - X @ b) / n
    J = n * gbar @ np.linalg.solve(S, gbar)
    df = Z.shape[1] - X.shape[1]
    return J, df, stats.chi2.sf(J, df)

one = np.ones(n)
X = np.column_stack([one, school])
Z_ols = X                                       # OLS: instruments = regressors
Z1 = np.column_stack([one, dist])               # just identified IV
Z2 = np.column_stack([one, dist, prog])         # over-identified

print(f"Simulated data, n = {n}, true return to schooling = 0.080")
print(f"{'estimator':<26} {'beta_s':>8} {'robust se':>10}")
rows = [("OLS", Z_ols, None), ("IV (distance)", Z1, None),
        ("2SLS (distance, programme)", Z2, None)]
for name, Z, _ in rows:
    W = np.linalg.inv(Z.T @ Z / n)              # 2SLS weighting
    b = gmm(logw, X, Z, W)
    V, S = robust_gmm_cov(logw, X, Z, W, b)
    print(f"{name:<26} {b[1]:8.4f} {np.sqrt(V[1, 1]):10.4f}")

# Two-step efficient GMM: W = S^{-1} from first-step residuals
b1 = gmm(logw, X, Z2, np.linalg.inv(Z2.T @ Z2 / n))
_, S1 = robust_gmm_cov(logw, X, Z2, np.eye(3), b1)
W2 = np.linalg.inv(S1)
b2 = gmm(logw, X, Z2, W2)
V2, S2 = robust_gmm_cov(logw, X, Z2, W2, b2)
print(f"{'two-step efficient GMM':<26} {b2[1]:8.4f} {np.sqrt(V2[1, 1]):10.4f}")
J, df, p = j_test(logw, X, Z2, b2, S1)
print(f"Hansen J (valid instruments)   = {J:6.3f}, df = {df}, p = {p:.3f}")

# An invalid 'instrument': household asset index correlated with ability
assets = rng.normal(size=n) + 0.5 * ability
Z3 = np.column_stack([one, dist, assets])
b1 = gmm(logw, X, Z3, np.linalg.inv(Z3.T @ Z3 / n))
_, S1 = robust_gmm_cov(logw, X, Z3, np.eye(3), b1)
b3 = gmm(logw, X, Z3, np.linalg.inv(S1))
J, df, p = j_test(logw, X, Z3, b3, S1)
print(f"Two-step GMM with assets: beta_s = {b3[1]:.4f}")
print(f"Hansen J (invalid instrument)  = {J:6.2f}, df = {df}, p = {p:.2g}")

# First-stage strength
coef, *_ = np.linalg.lstsq(Z2, school, rcond=None)
rss1 = np.sum((school - Z2 @ coef) ** 2)
rss0 = np.sum((school - school.mean()) ** 2)
Fstat = ((rss0 - rss1) / 2) / (rss1 / (n - 3))
print(f"First-stage F (distance, programme) = {Fstat:.1f}")
