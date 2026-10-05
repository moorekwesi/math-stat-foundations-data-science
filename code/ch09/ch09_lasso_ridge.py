# DEP 803 -- Chapter 9: Constrained Optimisation and Lagrange Multipliers
# Ridge and lasso: penalised form equals constrained form (diabetes data).
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.datasets import load_diabetes
from sklearn.linear_model import Lasso, lasso_path

X, y = load_diabetes(return_X_y=True)
X = (X - X.mean(0)) / X.std(0)
y = y - y.mean()
n, p = X.shape
rss = lambda b: np.sum((y - X @ b)**2) / (2 * n)
grad = lambda b: -X.T @ (y - X @ b) / n

# ---- ridge: penalised  min rss + (alpha/2)||b||^2 ------------------------
alpha = 5.0
b_pen = np.linalg.solve(X.T @ X / n + alpha * np.eye(p), X.T @ y / n)
t = b_pen @ b_pen                              # implied constraint level
con = {"type": "ineq", "fun": lambda b: t - b @ b, "jac": lambda b: -2 * b}
b_con = minimize(rss, np.zeros(p), jac=grad, method="SLSQP",
                 constraints=[con], tol=1e-14, options={"maxiter": 500}).x
mu = -(grad(b_con) @ b_con) / (2 * b_con @ b_con)  # from 0 = grad + 2 mu b
print("Ridge: alpha = %.1f  ->  t = ||b||^2 = %.3f" % (alpha, t))
print("  max |b_penalised - b_constrained| = %.2e" % np.abs(b_pen - b_con).max())
print("  recovered multiplier 2*mu = %.4f (equals alpha)" % (2 * mu))

# ---- lasso: penalised (sklearn) vs constrained ||b||_1 <= t -------------
alpha = 2.0
b_las = Lasso(alpha=alpha, fit_intercept=False, tol=1e-12,
              max_iter=100000).fit(X, y).coef_
t1 = np.abs(b_las).sum()
# split b = u - v with u, v >= 0 so that the l1 constraint is linear
f_uv = lambda z: rss(z[:p] - z[p:])
g_uv = lambda z: np.r_[grad(z[:p] - z[p:]), -grad(z[:p] - z[p:])]
con1 = {"type": "ineq", "fun": lambda z: t1 - z.sum(),
        "jac": lambda z: -np.ones(2 * p)}
z = minimize(f_uv, np.zeros(2 * p), jac=g_uv, method="SLSQP",
             bounds=[(0, None)] * (2 * p), constraints=[con1], tol=1e-14,
             options={"maxiter": 1000}).x
b_l1 = z[:p] - z[p:]
act = np.abs(b_l1) > 1e-6
mu1 = np.mean(-grad(b_l1)[act] / np.sign(b_l1[act]))
print("\nLasso: alpha = %.1f  ->  t = ||b||_1 = %.3f" % (alpha, t1))
print("  nonzero coefficients: sklearn %d, constrained %d" %
      ((np.abs(b_las) > 1e-6).sum(), act.sum()))
print("  max |b_sklearn - b_constrained| = %.2e" % np.abs(b_las - b_l1).max())
print("  multiplier from KKT on active set = %.4f (equals alpha)" % mu1)
print("  |grad_j| on zero coefficients <= alpha:",
      bool(np.all(np.abs(grad(b_l1)[~act]) <= alpha + 1e-4)))

# ---- figure: lasso path against the constraint level t ------------------
alphas, coefs, _ = lasso_path(X, y, n_alphas=200)
tt = np.abs(coefs).sum(0)
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(tt, coefs.T, lw=1.4)
ax.axvline(t1, ls="--", color="grey")
ax.annotate("t = %.1f (alpha = 2)" % t1, (t1 + 0.5, -20), fontsize=8)
ax.set_xlabel("constraint level t = ||b||_1")
ax.set_ylabel("standardised coefficient")
ax.set_title("Lasso as constrained least squares (diabetes data)",
             fontsize=10)
ax.grid(alpha=0.3)
plt.savefig("ch09_lasso_path.pdf", bbox_inches="tight")
