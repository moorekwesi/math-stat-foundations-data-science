# DEP 803 -- Chapter 7: Estimation Principles
# Complete separation: the logit MLE does not exist and optimisers drift away.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import optimize, special
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(803)
n = 40
x = np.sort(rng.normal(size=n))
y = (x > 0.1).astype(float)          # perfectly separated outcome
X = np.column_stack([np.ones(n), x])


def nll(b):
    eta = X @ b
    return np.sum(np.logaddexp(0, eta) - y * eta)


def grad(b):
    return X.T @ (special.expit(X @ b) - y)


print("Complete separation: y = 1 exactly when x > 0.1")
print(f"{'maxiter':>8} {'b0':>9} {'b1':>9} {'neg loglik':>11} {'success':>8}")
for it in (10, 50, 200, 1000):
    r = optimize.minimize(nll, np.zeros(2), jac=grad, method="BFGS",
                          options={"maxiter": it, "gtol": 1e-12})
    print(f"{it:>8} {r.x[0]:9.2f} {r.x[1]:9.2f} {r.fun:11.2e}"
          f" {str(r.success):>8}")
print(f"implied boundary -b0/b1 at maxiter = 1000: {-r.x[0] / r.x[1]:.4f}")
for C in (1.0, 100.0):
    sk = LogisticRegression(C=C).fit(x[:, None], y)
    print(f"sklearn L2-penalised, C = {C:>5}: b0 = {sk.intercept_[0]:6.2f},"
          f" b1 = {sk.coef_[0, 0]:6.2f}")
