# DEP 803 -- Chapter 9: Constrained Optimisation and Lagrange Multipliers
# Soft-margin linear SVM: solve the dual QP with SLSQP, compare with sklearn.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from sklearn.svm import SVC

rng = np.random.default_rng(803)
n = 40                                        # simulated two-class data
X = np.r_[rng.normal([1.0, 1.0], 0.7, (n // 2, 2)),
          rng.normal([-1.0, -0.8], 0.7, (n // 2, 2))]
y = np.r_[np.ones(n // 2), -np.ones(n // 2)]
C = 1.0
Q = (y[:, None] * X) @ (y[:, None] * X).T     # Q_ij = y_i y_j x_i'x_j

# dual: max sum(a) - a'Qa/2  s.t.  0 <= a <= C,  y'a = 0
obj = lambda a: 0.5 * a @ Q @ a - a.sum()
jac = lambda a: Q @ a - 1
res = minimize(obj, np.zeros(n), jac=jac, method="SLSQP",
               bounds=[(0, C)] * n, tol=1e-12, options={"maxiter": 500},
               constraints=[{"type": "eq", "fun": lambda a: y @ a,
                             "jac": lambda a: y}])
a = np.where(res.x < 1e-7, 0.0, res.x)
w = (a * y) @ X                               # primal from stationarity
free = (a > 1e-6) & (a < C - 1e-6)            # margin support vectors
b = np.mean(y[free] - X[free] @ w)

svc = SVC(kernel="linear", C=C, tol=1e-8).fit(X, y)
print("Dual SLSQP:  w = %s  b = %.4f" % (np.round(w, 4), b))
print("sklearn SVC: w = %s  b = %.4f" % (np.round(svc.coef_[0], 4),
                                          svc.intercept_[0]))
print("support vectors: dual %d (on margin %d, a = C %d), sklearn %d" %
      ((a > 1e-6).sum(), free.sum(), (a >= C - 1e-6).sum(),
       len(svc.support_)))
xi = np.maximum(0, 1 - y * (X @ w + b))       # slack variables
primal = 0.5 * w @ w + C * xi.sum()
print("primal value %.5f   dual value %.5f   gap %.1e" %
      (primal, -res.fun, primal + res.fun))
print("complementary slackness: max (C - a_i) xi_i = %.1e" %
      np.max((C - a) * xi))

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.scatter(*X[y > 0].T, c="C0", s=18, label="class +1")
ax.scatter(*X[y < 0].T, c="C1", s=18, label="class -1")
ax.scatter(*X[a > 1e-6].T, s=70, facecolors="none", edgecolors="k",
           label="support vectors")
g = np.linspace(-3, 3, 50)
for off, ls in [(0, "-"), (1, "--"), (-1, "--")]:
    ax.plot(g, (off - b - w[0] * g) / w[1], "k", ls=ls, lw=1)
ax.set_xlim(-3, 3)
ax.set_ylim(-3, 3)
ax.set_xlabel("feature 1")
ax.set_ylabel("feature 2")
ax.set_title("Soft-margin SVM from its dual (C = 1)", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
plt.savefig("ch09_svm_dual.pdf", bbox_inches="tight")
