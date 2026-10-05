# DEP 803 -- Chapter 5: Common Distributions in Data Science and Econometrics
# Multivariate normal: Cholesky simulation, conditional moments, Mahalanobis distance
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)
# (log income, years of schooling, household size) -- simulated
mu = np.array([9.4, 8.0, 5.0])
Sigma = np.array([[0.64, 1.20, -0.30],
                  [1.20, 9.00, -0.90],
                  [-0.30, -0.90, 4.00]])
L = np.linalg.cholesky(Sigma)                 # Sigma = L L^T
n = 400_000
X = mu + rng.standard_normal((n, 3)) @ L.T    # rows are N(mu, Sigma) draws
print("max |sample cov - Sigma| =", np.abs(np.cov(X.T) - Sigma).max().round(4))

# conditional distribution of X1 (log income) given X2 = 12, X3 = 4
a = np.array([12.0, 4.0])
S11, S12, S22 = Sigma[0, 0], Sigma[0, 1:], Sigma[1:, 1:]
w = np.linalg.solve(S22, S12)                 # regression coefficients
cmean = mu[0] + w @ (a - mu[1:])
cvar = S11 - S12 @ w                          # Schur complement
near = np.all(np.abs(X[:, 1:] - a) < 0.15, axis=1)
print(f"E[X1 | X2=12, X3=4]: formula {cmean:.4f}, "
      f"simulation {X[near, 0].mean():.4f} ({near.sum()} draws)")
print(f"Var[X1 | ...]      : formula {cvar:.4f}, simulation "
      f"{X[near, 0].var():.4f}")
print("population regression coefficients Sigma22^-1 Sigma21 =", w.round(4))

# Mahalanobis distance: D^2 ~ chi-square(3)
d2 = np.einsum("ij,ij->i", X - mu, np.linalg.solve(Sigma, (X - mu).T).T)
print(f"P(D^2 > chi2_3(0.95)) = {(d2 > stats.chi2(3).ppf(0.95)).mean():.4f}")
x0 = np.array([11.0, 6.0, 5.0])
d0 = np.sqrt((x0 - mu) @ np.linalg.solve(Sigma, x0 - mu))
print(f"household (11, 6, 5): Euclidean dist {np.linalg.norm(x0 - mu):.3f}, "
      f"Mahalanobis {d0:.3f}")

# figure: (log income, schooling) scatter with Mahalanobis contours
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.scatter(X[:3000, 1], X[:3000, 0], s=3, alpha=0.3, color="#1f4e79")
S2 = Sigma[np.ix_([1, 0], [1, 0])]
vals, vecs = np.linalg.eigh(S2)
th = np.linspace(0, 2 * np.pi, 200)
for c in [1, 2, 3]:
    pts = c * vecs @ np.diag(np.sqrt(vals)) @ np.vstack([np.cos(th), np.sin(th)])
    ax.plot(mu[1] + pts[0], mu[0] + pts[1], color="#cc9900", lw=1.5)
ax.scatter([x0[1]], [x0[0]], color="#b22222", zorder=3, label="household x0")
xs = np.linspace(-2, 18, 50)
ax.plot(xs, mu[0] + S2[0, 1] / S2[0, 0] * (xs - mu[1]), "k--", lw=1,
        label="E[log income | schooling]")
ax.set_xlabel("years of schooling (simulated)")
ax.set_ylabel("log income (simulated)")
ax.legend(frameon=False, fontsize=8, loc="lower right")
plt.savefig("ch05_mvn_contours.pdf", bbox_inches="tight")
