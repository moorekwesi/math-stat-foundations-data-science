# DEP 803 -- Chapter 13: Problem Set Clinic and Applied Foundations Review
# Mincer regression on a simulated labour force survey: QR, FWL, HC1 by hand.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np


def simulate_lfs(rng, n=4000, n_reg=10):
    """Simulated labour force survey (NOT real data for any country)."""
    region = rng.integers(0, n_reg, n)
    reg_fe = np.linspace(-0.3, 0.3, n_reg)           # regional wage levels
    reg_edu = np.linspace(-2.0, 2.0, n_reg)          # regional schooling gaps
    female = (rng.random(n) < 0.5).astype(float)
    age = rng.integers(18, 61, n)
    educ = np.clip(np.round(9 + reg_edu[region] + 3.5 * rng.standard_normal(n)),
                   0, 20)
    exper = np.maximum(age - educ - 6, 0)
    sd = 0.10 + 0.045 * educ                         # heteroskedastic errors
    logw = (1.0 + reg_fe[region] + 0.10 * educ + 0.035 * exper
            - 0.0006 * exper**2 - 0.12 * female + sd * rng.standard_normal(n))
    return logw, educ, exper, female, region


def ols_qr(X, y):
    """OLS via thin QR: returns beta, residuals and R."""
    Q, R = np.linalg.qr(X)
    beta = np.linalg.solve(R, Q.T @ y)
    return beta, y - X @ beta, R


rng = np.random.default_rng(803)
logw, educ, exper, female, region = simulate_lfs(rng)
n = logw.size
D = (region[:, None] == np.arange(1, 10)).astype(float)   # 9 region dummies
X = np.column_stack([np.ones(n), educ, exper, exper**2, female, D])
k = X.shape[1]
beta, e, R = ols_qr(X, logw)

# Classical and HC1 standard errors, using (X'X)^{-1} = R^{-1} R^{-T}
Rinv = np.linalg.solve(R, np.eye(k))
XtX_inv = Rinv @ Rinv.T
s2 = e @ e / (n - k)
se_cl = np.sqrt(np.diag(s2 * XtX_inv))
meat = (X * e[:, None]**2).T @ X
V_hc1 = n / (n - k) * XtX_inv @ meat @ XtX_inv
se_hc1 = np.sqrt(np.diag(V_hc1))

names = ["const", "educ", "exper", "exper^2", "female"]
print(f"n = {n}, k = {k} (incl. 9 region dummies), cond(X) = "
      f"{np.linalg.cond(X):.2e}")
print(f"{'':<9}{'coef':>10}{'se(class)':>11}{'se(HC1)':>10}{'ratio':>7}")
for j, nm in enumerate(names):
    print(f"{nm:<9}{beta[j]:10.5f}{se_cl[j]:11.5f}{se_hc1[j]:10.5f}"
          f"{se_hc1[j] / se_cl[j]:7.2f}")
b_ls = np.linalg.lstsq(X, logw, rcond=None)[0]
print("agrees with lstsq:", np.allclose(beta, b_ls))

# Frisch-Waugh-Lovell: partial out controls Z (all columns except educ)
Z = np.delete(X, 1, axis=1)
_, e_educ, _ = ols_qr(Z, educ)
_, e_logw, _ = ols_qr(Z, logw)
b_fwl = (e_educ @ e_logw) / (e_educ @ e_educ)
# Within (demeaning by region) transformation = FWL for the region dummies
def demean(v):
    means = np.bincount(region, weights=v) / np.bincount(region)
    return v - means[region]
W = np.column_stack([demean(c) for c in (educ, exper, exper**2, female)])
b_within = np.linalg.lstsq(W, demean(logw), rcond=None)[0]
print(f"\nreturn to schooling: full OLS {beta[1]:.6f}, FWL {b_fwl:.6f}, "
      f"within {b_within[0]:.6f}")
print(f"no region FE        : {ols_qr(X[:, :5], logw)[0][1]:.6f}")
pct = 100 * (np.exp(beta[1]) - 1)
lo, hi = beta[1] - 1.96 * se_hc1[1], beta[1] + 1.96 * se_hc1[1]
print(f"one more year of school: +{pct:.2f}% wage; 95% HC1 CI for beta "
      f"[{lo:.4f}, {hi:.4f}]")
