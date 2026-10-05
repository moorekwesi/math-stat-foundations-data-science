# DEP 803 -- Chapter 12: Matrix Calculus, Regularisation and Bias-Variance
# Lasso vs elastic net with groups of highly correlated predictors.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(803)
n = 100
# two latent drivers (e.g. 'labour market', 'prices'), three noisy proxies each
z = rng.normal(size=(n, 2))
X = np.column_stack([z[:, [0]] + 0.1 * rng.normal(size=(n, 3)),
                     z[:, [1]] + 0.1 * rng.normal(size=(n, 3)),
                     rng.normal(size=(n, 4))])          # 4 irrelevant
X = StandardScaler().fit_transform(X)
y = 3 * z[:, 0] - 2 * z[:, 1] + rng.normal(size=n)
y = y - y.mean()
print("corr within group 1:", np.round(np.corrcoef(X[:, :3].T)[0, 1:], 3))

las = Lasso(alpha=0.5, fit_intercept=False).fit(X, y)
en = ElasticNet(alpha=0.6, l1_ratio=0.2, fit_intercept=False).fit(X, y)
print(f"{'':10}" + "".join(f"{'x' + str(j + 1):>7}" for j in range(10)))
print(f"{'lasso':10}" + "".join(f"{c:>7.2f}" for c in las.coef_))
print(f"{'enet':10}" + "".join(f"{c:>7.2f}" for c in en.coef_))
for name, m in [("lasso", las), ("enet", en)]:
    g1, g2 = m.coef_[:3], m.coef_[3:6]
    print(f"{name:6}: nonzero in group 1 = {np.sum(g1 != 0)}, "
          f"group 2 = {np.sum(g2 != 0)}, noise = {np.sum(m.coef_[6:] != 0)};"
          f" group sums {g1.sum():.2f}, {g2.sum():.2f}")
