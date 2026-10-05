# DEP 803 -- Chapter 5: Common Distributions in Data Science and Econometrics
# Gaussian regression likelihood: numerical MLE equals least squares
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import optimize
from sklearn.datasets import load_diabetes

X0, y = load_diabetes(return_X_y=True)        # bundled scikit-learn data
X = np.column_stack([np.ones(len(y)), X0[:, [2, 3, 8]]])   # bmi, bp, s5
n, k = X.shape


def negloglik(theta):
    """Negative Gaussian log-likelihood; theta = (beta, log sigma)."""
    beta, log_s = theta[:-1], theta[-1]
    r = y - X @ beta
    return (n * log_s + 0.5 * n * np.log(2 * np.pi)
            + 0.5 * (r @ r) * np.exp(-2 * log_s))


start = np.r_[np.zeros(k), np.log(y.std())]
res = optimize.minimize(negloglik, start, method="Powell",
                        options={"xtol": 1e-10, "ftol": 1e-14, "maxiter": 50_000})
res = optimize.minimize(negloglik, res.x, method="BFGS")   # polish
beta_ml, s_ml = res.x[:-1], np.exp(res.x[-1])
beta_ls, *_ = np.linalg.lstsq(X, y, rcond=None)
rss = np.sum((y - X @ beta_ls) ** 2)

print(f"{'coef':8s}{'MLE (numeric)':>14s}{'OLS (lstsq)':>14s}")
for name, a, b in zip(["const", "bmi", "bp", "s5"], beta_ml, beta_ls):
    print(f"{name:8s}{a:14.4f}{b:14.4f}")
print(f"sigma_ML = {s_ml:.4f},  sqrt(RSS/n) = {np.sqrt(rss / n):.4f},  "
      f"sqrt(RSS/(n-k)) = {np.sqrt(rss / (n - k)):.4f}")
print(f"max loglik = {-res.fun:.3f}; formula -n/2 (log(2 pi RSS/n) + 1) = "
      f"{-n / 2 * (np.log(2 * np.pi * rss / n) + 1):.3f}")
