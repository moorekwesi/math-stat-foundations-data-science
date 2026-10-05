# DEP 803 -- Chapter 11: Resampling: Bootstrap, Permutation Tests, Intervals
# LOO and K-fold cross-validation for ridge regression on load_diabetes.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

rng = np.random.default_rng(803)
X, y = load_diabetes(return_X_y=True)
X = PolynomialFeatures(2, include_bias=False).fit_transform(X)   # 65 features
X = StandardScaler().fit_transform(X)
y = y - y.mean()
n, p = X.shape
lams = np.logspace(-1, 4, 26)


def ridge_fit(X, y, lam):
    return np.linalg.solve(X.T @ X + lam * np.eye(X.shape[1]), X.T @ y)


def loo_shortcut(lam):
    """Exact LOO MSE for ridge: mean of (e_i / (1 - h_ii))^2."""
    H = X @ np.linalg.solve(X.T @ X + lam * np.eye(p), X.T)
    e = y - H @ y
    return np.mean((e / (1 - np.diag(H)))**2), np.mean(e**2)


def kfold(lam, K, perm):
    folds = np.array_split(perm, K)
    err = 0.0
    for f in folds:
        tr = np.setdiff1d(perm, f)
        mu = X[tr].mean(0)
        b = ridge_fit(X[tr] - mu, y[tr] - y[tr].mean(), lam)
        err += np.sum((y[f] - y[tr].mean() - (X[f] - mu) @ b)**2)
    return err / n


# Check the shortcut against brute-force LOO at one value of lambda
lam0 = 100.0
bf = np.mean([(y[i] - X[i] @ ridge_fit(np.delete(X, i, 0), np.delete(y, i),
                                         lam0))**2 for i in range(n)])
print(f"n = {n}, p = {p}; LOO at lambda = {lam0:g}: shortcut = "
      f"{loo_shortcut(lam0)[0]:.2f}, brute force = {bf:.2f}")

train, loo = np.array([loo_shortcut(l)[::-1] for l in lams]).T
cv5 = np.array([[kfold(l, 5, rng.permutation(n)) for l in lams]
                for _ in range(10)])                    # 10 random splits
cv10 = np.array([kfold(l, 10, rng.permutation(n)) for l in lams])
print(f"{'lambda':>9}{'train MSE':>11}{'LOO':>9}{'10-fold':>9}"
      f"{'5-fold':>9}{'sd(5-fold)':>12}")
for i in range(0, len(lams), 5):
    print(f"{lams[i]:9.1f}{train[i]:11.1f}{loo[i]:9.1f}{cv10[i]:9.1f}"
          f"{cv5[:, i].mean():9.1f}{cv5[:, i].std():12.1f}")
k = np.argmin(loo)
print(f"lambda chosen by LOO = {lams[k]:.1f} (LOO MSE {loo[k]:.1f}); "
      f"by 10-fold = {lams[np.argmin(cv10)]:.1f}")

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.semilogx(lams, train, "k--", label="training MSE")
ax.semilogx(lams, loo, color="firebrick", label="LOO (exact shortcut)")
for r in range(10):
    ax.semilogx(lams, cv5[r], color="steelblue", lw=0.6, alpha=0.6,
                label="5-fold, 10 random splits" if r == 0 else None)
ax.set_xlabel(r"ridge penalty $\lambda$")
ax.set_ylabel("mean squared prediction error")
ax.legend(frameon=False)
plt.savefig("ch11_cv.pdf", bbox_inches="tight")
