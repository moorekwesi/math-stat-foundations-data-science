# DEP 803 -- Chapter 12: Matrix Calculus, Regularisation and Bias-Variance
# Logistic regression: analytic gradient/Hessian, Newton-Raphson, sklearn check.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(803)

# simulated loan applications: default (1) vs repay (0)
n = 2000
Z = rng.normal(size=(n, 3))          # debt ratio, income, credit history
X = np.column_stack([np.ones(n), Z])
beta_true = np.array([-1.0, 1.2, -0.8, -0.5])
prob = 1 / (1 + np.exp(-X @ beta_true))
y = rng.binomial(1, prob)


def negloglik(b):
    eta = X @ b
    return np.sum(np.logaddexp(0, eta) - y * eta)


def grad_hess(b):
    p = 1 / (1 + np.exp(-X @ b))
    g = X.T @ (p - y)                              # gradient
    H = X.T @ (X * (p * (1 - p))[:, None])         # Hessian X'WX
    return g, H


# gradient check at a random point
b0 = rng.normal(size=4) * 0.1
g0, _ = grad_hess(b0)
h = 1e-6
g_num = np.array([(negloglik(b0 + h * e) - negloglik(b0 - h * e)) / (2 * h)
                  for e in np.eye(4)])
print(f"gradient check: max abs diff = {np.abs(g0 - g_num).max():.2e}")

# Newton-Raphson (= iteratively reweighted least squares)
b = np.zeros(4)
print(f"{'iter':>4} {'neg loglik':>12} {'||grad||':>11}")
for it in range(8):
    g, H = grad_hess(b)
    print(f"{it:>4} {negloglik(b):>12.5f} {np.linalg.norm(g):>11.3e}")
    if np.linalg.norm(g) < 1e-10:
        break
    b = b - np.linalg.solve(H, g)

se = np.sqrt(np.diag(np.linalg.inv(grad_hess(b)[1])))
skl = LogisticRegression(penalty="none", tol=1e-10, max_iter=1000)
skl.fit(Z, y)
print("Newton beta :", np.round(b, 4))
print("std errors  :", np.round(se, 4))
print("sklearn     :", np.round(np.r_[skl.intercept_, skl.coef_.ravel()], 4))
