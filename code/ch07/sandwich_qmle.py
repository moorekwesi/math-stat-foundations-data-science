# DEP 803 -- Chapter 7: Estimation Principles
# Poisson quasi-MLE with overdispersed counts: model vs sandwich standard errors.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
n, R = 400, 3000
g_true = np.array([0.8, 0.3, -0.35])


def poisson_fit(X, y):
    """Poisson MLE by Newton-Raphson; returns b, A = -Hessian, B = sum s s'."""
    b = np.zeros(X.shape[1])
    b[0] = np.log(y.mean())
    for _ in range(50):
        mu = np.exp(X @ b)
        A = (X * mu[:, None]).T @ X
        step = np.linalg.solve(A, X.T @ (y - mu))
        b += step
        if np.max(np.abs(step)) < 1e-10:
            break
    mu = np.exp(X @ b)
    A = (X * mu[:, None]).T @ X
    s = X * (y - mu)[:, None]                     # individual scores
    return b, A, s.T @ s


def simulate(alpha):
    """Negative binomial counts with E[y|x] = exp(x'g), Var = mu + alpha mu^2."""
    X = np.column_stack([np.ones(n), rng.binomial(1, 0.4, n),
                         rng.uniform(0, 2, n)])
    mu = np.exp(X @ g_true)
    lam = mu * rng.gamma(1 / alpha, alpha, n) if alpha > 0 else mu
    return X, rng.poisson(lam)


print(f"Poisson (quasi-)MLE, n = {n}, R = {R}; coverage of 95% intervals"
      " for the fee slope")
print(f"{'alpha':>6} {'MC sd':>8} {'model se':>9} {'sandwich':>9}"
      f" {'cov model':>10} {'cov sand':>9}")
for alpha in (0.0, 0.5, 1.5):
    est, se_m, se_s = np.empty(R), np.empty(R), np.empty(R)
    for r in range(R):
        X, y = simulate(alpha)
        b, A, Bm = poisson_fit(X, y)
        Ai = np.linalg.inv(A)
        est[r] = b[2]
        se_m[r] = np.sqrt(Ai[2, 2])
        se_s[r] = np.sqrt((Ai @ Bm @ Ai)[2, 2])
    err = np.abs(est - g_true[2])
    print(f"{alpha:6.1f} {est.std():8.4f} {se_m.mean():9.4f}"
          f" {se_s.mean():9.4f} {np.mean(err < 1.96 * se_m):10.3f}"
          f" {np.mean(err < 1.96 * se_s):9.3f}")
print(f"mean of fee slope estimates (alpha = 1.5): {est.mean():.4f}"
      f" (true {g_true[2]})")
