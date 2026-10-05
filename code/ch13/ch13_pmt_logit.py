# DEP 803 -- Chapter 13: Problem Set Clinic and Applied Foundations Review
# Proxy-means test: PCA asset index and a logit fitted by Newton/IRLS by hand.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

ASSETS = ["radio", "tv", "fridge", "phone", "motorbike", "floor", "electric",
          "water", "toilet", "gas"]


def simulate_households(rng, n=4000):
    """Simulated household survey: assets, size, rural, poverty status."""
    rural = (rng.random(n) < 0.55).astype(float)
    hhsize = np.clip(rng.poisson(4.5 + 1.0 * rural), 1, 14).astype(float)
    welfare = rng.standard_normal(n) - 0.6 * rural   # latent living standard
    a = np.array([0.8, -0.2, -1.0, 1.2, -0.8, 0.0, 0.3, -0.4, -0.3, -1.3])
    b = np.array([0.5, 1.1, 1.2, 0.6, 0.6, 0.9, 1.0, 0.9, 1.0, 1.1])
    assets = (a + np.outer(welfare, b)
              + rng.standard_normal((n, a.size)) > 0).astype(float)
    logc = (7.0 + 0.55 * welfare - 0.07 * hhsize
            + 0.35 * rng.standard_normal(n))       # log consumption per head
    poor = (logc < np.quantile(logc, 0.30)).astype(float)
    return assets, hhsize, rural, poor


def irls_logit(X, y, tol=1e-10, max_iter=25, verbose=True):
    """Newton-Raphson (= IRLS) for the logistic MLE."""
    beta = np.zeros(X.shape[1])
    for it in range(max_iter):
        eta = X @ beta
        p = 1 / (1 + np.exp(-eta))
        w = p * (1 - p)
        grad = X.T @ (y - p)
        H = (X * w[:, None]).T @ X                  # minus the Hessian
        step = np.linalg.solve(H, grad)
        beta = beta + step
        ll = np.sum(y * eta - np.logaddexp(0, eta))
        if verbose:
            print(f"  iter {it}: loglik(before step) = {ll:10.4f}, "
                  f"|step| = {np.linalg.norm(step):.2e}")
        if np.linalg.norm(step) < tol:
            break
    return beta, np.linalg.inv(H)


rng = np.random.default_rng(803)
assets, hhsize, rural, poor = simulate_households(rng)
n = poor.size
train = rng.random(n) < 0.7
test = ~train

# (a) PCA asset index (Filmer-Pritchett style): first PC of standardised assets
mu, sd = assets[train].mean(0), assets[train].std(0)
Zs = (assets - mu) / sd
_, s, Vt = np.linalg.svd(Zs[train], full_matrices=False)
w1 = Vt[0] * np.sign(Vt[0].sum())                   # orient: more assets = richer
index = Zs @ w1
print(f"PC1 explains {s[0]**2 / np.sum(s**2):.1%} of standardised asset variance")
for row in (slice(0, 5), slice(5, 10)):
    print("  weights:", " ".join(f"{a}={w:.2f}"
                                 for a, w in zip(ASSETS[row], w1[row])))

# (b) logit proxy-means test by IRLS
X = np.column_stack([np.ones(n), assets, hhsize, rural])
print("IRLS iterations:")
beta, cov = irls_logit(X[train], poor[train])
sk = LogisticRegression(penalty="none", tol=1e-12, max_iter=5000)
sk.fit(X[train, 1:], poor[train])
b_sk = np.r_[sk.intercept_, sk.coef_.ravel()]
print(f"max |beta_IRLS - beta_sklearn| = {np.max(np.abs(beta - b_sk)):.1e}")
se = np.sqrt(np.diag(cov))
for nm, j in [("tv", 2), ("fridge", 3), ("hhsize", 11), ("rural", 12)]:
    print(f"  {nm:<7} coef {beta[j]:7.3f}  se {se[j]:.3f}  "
          f"odds ratio {np.exp(beta[j]):.2f}")

p_test = 1 / (1 + np.exp(-X[test] @ beta))
print(f"test AUC: PCA index {roc_auc_score(poor[test], -index[test]):.3f}, "
      f"logit PMT {roc_auc_score(poor[test], p_test):.3f}")
