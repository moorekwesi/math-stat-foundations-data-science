# DEP 803 -- Chapter 7: Estimation Principles
# MAP with a Gaussian prior is ridge regression (diabetes data bundled in sklearn).
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Ridge

X, y = load_diabetes(return_X_y=True)
X = (X - X.mean(0)) / X.std(0)           # standardise the 10 predictors
y = y - y.mean()
n, p = X.shape
sigma2 = 54.0**2                         # noise variance, treated as known
print(f"Diabetes data: n = {n}, p = {p}; sigma = 54 treated as known")
print(f"{'prior sd tau':>13} {'lambda':>9} {'||b_MAP - b_ridge||':>20}"
      f" {'||b_MAP||':>10}")
for tau in (100.0, 10.0, 3.0, 1.0):
    lam = sigma2 / tau**2
    # posterior: b | y ~ N(m, V), V = (X'X/s2 + I/tau2)^{-1}, m = V X'y/s2
    V = np.linalg.inv(X.T @ X / sigma2 + np.eye(p) / tau**2)
    m = V @ X.T @ y / sigma2
    br = Ridge(alpha=lam, fit_intercept=False).fit(X, y).coef_
    print(f"{tau:13.1f} {lam:9.1f} {np.linalg.norm(m - br):20.2e}"
          f" {np.linalg.norm(m):10.2f}")
ols = np.linalg.lstsq(X, y, rcond=None)[0]
print(f"{'(flat prior)':>13} {0.0:9.1f} {'OLS':>20} {np.linalg.norm(ols):10.2f}")
tau = 3.0
V = np.linalg.inv(X.T @ X / sigma2 + np.eye(p) / tau**2)
m = V @ X.T @ y / sigma2
print("Posterior for the bmi coefficient (tau = 3): mean %.2f, sd %.2f"
      % (m[2], np.sqrt(V[2, 2])))
