# DEP 803 -- Chapter 8: Optimisation
# Newton-Raphson / IRLS for a logit of programme take-up (simulated data)
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

rng = np.random.default_rng(803)
n = 5000
# simulated beneficiary survey: does the household take up a subsidy?
loginc = rng.normal(7.5, 0.8, n)          # log monthly income
female = rng.binomial(1, 0.45, n)         # female-headed household
dist = rng.gamma(2.0, 3.0, n)             # km to the district office
X = np.column_stack([np.ones(n), loginc - 7.5, female, dist])
names = ["const", "loginc", "female", "dist_km"]
beta_true = np.array([0.8, -0.9, 0.5, -0.12])
y = (rng.uniform(size=n) < 1 / (1 + np.exp(-X @ beta_true))).astype(float)


def newton(b, tol=1e-10, maxit=50):
    hist = []
    for k in range(maxit):
        p = 1 / (1 + np.exp(-X @ b))
        g = X.T @ (p - y)                       # gradient of the NLL
        hist.append(np.linalg.norm(g))
        W = p * (1 - p)
        H = (X * W[:, None]).T @ X              # Hessian X'WX
        if hist[-1] < tol:
            break
        b = b - np.linalg.solve(H, g)           # Newton step = IRLS update
    return b, H, hist


b_hat, H, hist = newton(np.zeros(4))
print("Newton iterations: ||gradient||_k and ratio g_k / g_(k-1)^2")
for k, gk in enumerate(hist):
    ratio = f"{gk / hist[k - 1]**2:9.2e}" if 0 < k and hist[k - 1] < 1 else ""
    print(f"  k = {k}: {gk:10.3e}   {ratio}")
se = np.sqrt(np.diag(np.linalg.inv(H)))
sk = LogisticRegression(penalty="none", fit_intercept=False, tol=1e-10,
                        max_iter=1000).fit(X, y)
print(f"{'':>9}{'truth':>8}{'Newton':>9}{'se':>8}{'sklearn':>9}")
for j in range(4):
    print(f"{names[j]:>9}{beta_true[j]:>8.3f}{b_hat[j]:>9.4f}{se[j]:>8.4f}"
          f"{sk.coef_[0, j]:>9.4f}")

# gradient descent on the same objective, step 1/L with L = lambda_max(X'X)/4
L = np.linalg.eigvalsh(X.T @ X).max() / 4
b, gd_hist = np.zeros(4), []
for k in range(5000):
    g = X.T @ (1 / (1 + np.exp(-X @ b)) - y)
    gd_hist.append(np.linalg.norm(g))
    b -= g / L
print(f"GD with step 1/L: ||grad|| after 5000 iters = {gd_hist[-1]:.2e}")
print(f"max |b_GD - b_Newton| = {np.abs(b - b_hat).max():.2e}")

plt.figure(figsize=(6, 3.8))
plt.semilogy(gd_hist, label="gradient descent (step 1/L)")
plt.semilogy(range(len(hist)), hist, "o-", label="Newton / IRLS")
plt.xscale("symlog", linthresh=10)
plt.gca().set(xlabel="iteration (symlog scale)", ylabel="norm of gradient",
              title="Logit take-up model: Newton vs gradient descent")
plt.legend(frameon=False)
plt.savefig("ch08_newton_conv.pdf", bbox_inches="tight")
