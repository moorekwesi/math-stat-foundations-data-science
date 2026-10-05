# DEP 803 -- Chapter 5: Common Distributions in Data Science and Econometrics
# Logistic and Poisson regression as GLMs fitted by IRLS, checked with scikit-learn
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from sklearn.linear_model import LogisticRegression, PoissonRegressor

rng = np.random.default_rng(803)
n = 4000
# simulated households: years of schooling of head, distance to clinic (km)
school = rng.uniform(0, 16, n)
dist = rng.exponential(5.0, n)
X = np.column_stack([np.ones(n), school, dist])

# binary outcome: child enrolled in secondary school (logit link)
beta_logit = np.array([-1.0, 0.25, -0.10])
y_bin = rng.binomial(1, 1 / (1 + np.exp(-X @ beta_logit)))
# count outcome: clinic visits per year (log link)
beta_pois = np.array([1.2, 0.03, -0.08])
y_cnt = rng.poisson(np.exp(X @ beta_pois))


def irls(X, y, family, iters=25):
    """Iteratively reweighted least squares for canonical-link GLMs."""
    b = np.zeros(X.shape[1])
    for it in range(iters):
        eta = X @ b
        mu = 1 / (1 + np.exp(-eta)) if family == "logit" else np.exp(eta)
        w = mu * (1 - mu) if family == "logit" else mu    # Var(y) = A''(eta)
        z = eta + (y - mu) / w                             # working response
        b_new = np.linalg.solve(X.T @ (w[:, None] * X), X.T @ (w * z))
        if np.max(np.abs(b_new - b)) < 1e-10:
            return b_new, it + 1, w
        b = b_new
    return b, iters, w


for fam, y, beta in [("logit", y_bin, beta_logit), ("log", y_cnt, beta_pois)]:
    b, its, w = irls(X, y, fam)
    se = np.sqrt(np.diag(np.linalg.inv(X.T @ (w[:, None] * X))))
    if fam == "logit":
        sk = LogisticRegression(penalty="none", tol=1e-10, max_iter=1000)
        sk.fit(X[:, 1:], y)
    else:
        sk = PoissonRegressor(alpha=0.0, tol=1e-10, max_iter=1000)
        sk.fit(X[:, 1:], y)
    sk_b = np.r_[np.ravel(sk.intercept_), np.ravel(sk.coef_)]
    print(f"{fam} link: IRLS converged in {its} iterations")
    print("   true   " + " ".join(f"{v:9.4f}" for v in beta))
    print("   IRLS   " + " ".join(f"{v:9.4f}" for v in b))
    print("   s.e.   " + " ".join(f"{v:9.4f}" for v in se))
    print("   sklearn" + " ".join(f"{v:9.4f}" for v in sk_b))
b_logit = irls(X, y_bin, "logit")[0]
print(f"odds ratio for +4 years of schooling = {np.exp(4 * b_logit[1]):.3f}")
b_pois = irls(X, y_cnt, "log")[0]
print(f"rate ratio for +10 km distance = {np.exp(10 * b_pois[2]):.3f}")
