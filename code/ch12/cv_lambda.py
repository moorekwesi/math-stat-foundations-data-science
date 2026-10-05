# DEP 803 -- Chapter 12: Matrix Calculus, Regularisation and Bias-Variance
# K-fold cross-validation for the lasso penalty, with the one-standard-error rule.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold

rng = np.random.default_rng(803)
n, p = 150, 60                             # simulated district-level panel
X = rng.normal(size=(n, p))
beta = np.zeros(p)
beta[:8] = [1.5, -1.2, 1.0, 0.8, -0.6, 0.5, 0.4, -0.3]
y = X @ beta + 1.0 * rng.normal(size=n)

alphas = np.logspace(0, -3, 40)
K = 10
folds = KFold(n_splits=K, shuffle=True, random_state=803)
cv_err = np.zeros((K, alphas.size))
for k, (tr, va) in enumerate(folds.split(X)):
    mu, sd = X[tr].mean(0), X[tr].std(0)   # standardise using training fold
    Xtr, Xva = (X[tr] - mu) / sd, (X[va] - mu) / sd
    for a, alpha in enumerate(alphas):
        m = Lasso(alpha=alpha, max_iter=20000).fit(Xtr, y[tr])
        cv_err[k, a] = np.mean((y[va] - m.predict(Xva)) ** 2)
cv_mean = cv_err.mean(0)
cv_se = cv_err.std(0, ddof=1) / np.sqrt(K)
i_min = np.argmin(cv_mean)
ok = cv_mean <= cv_mean[i_min] + cv_se[i_min]
i_1se = np.flatnonzero(ok)[0]              # largest alpha within one SE
Xs = (X - X.mean(0)) / X.std(0)
for lab, i in [("alpha_min", i_min), ("alpha_1se", i_1se)]:
    m = Lasso(alpha=alphas[i], max_iter=20000).fit(Xs, y)
    print(f"{lab}: {alphas[i]:.4f}  CV MSE = {cv_mean[i]:.3f} "
          f"(se {cv_se[i]:.3f})  nonzero = {np.sum(m.coef_ != 0)}, "
          f"true found = {np.sum(m.coef_[:8] != 0)}/8")
auto = LassoCV(alphas=alphas, cv=folds, max_iter=20000).fit(Xs, y)
print(f"LassoCV alpha = {auto.alpha_:.4f} (same folds; standardised once)")

plt.figure(figsize=(6, 3.8))
plt.errorbar(np.log10(alphas), cv_mean, yerr=cv_se, fmt="o", ms=3,
             capsize=2, label="10-fold CV MSE +/- 1 se")
plt.axvline(np.log10(alphas[i_min]), color="C1", ls="--", label="minimum")
plt.axvline(np.log10(alphas[i_1se]), color="C2", ls=":", lw=2,
            label="one-standard-error rule")
plt.xlabel(r"$\log_{10}\alpha$ (larger = simpler model)")
plt.ylabel("cross-validated MSE")
plt.title("Choosing the lasso penalty by K-fold CV")
plt.legend(fontsize=9)
plt.grid(alpha=0.3)
plt.savefig("ch12_cv_curve.pdf", bbox_inches="tight")
