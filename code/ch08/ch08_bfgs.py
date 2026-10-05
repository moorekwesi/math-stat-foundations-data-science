# DEP 803 -- Chapter 8: Optimisation
# Comparing scipy.optimize.minimize methods: iterations, evaluations, time
# (c) 2026 Stephen E. Moore. MIT Licence.
import time
import numpy as np
from scipy.optimize import minimize, rosen, rosen_der, rosen_hess

rng = np.random.default_rng(803)
n, p = 10000, 20
X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
beta = rng.normal(scale=0.3, size=p)
y = (rng.uniform(size=n) < 1 / (1 + np.exp(-X @ beta))).astype(float)


def f(b):
    z = X @ b
    return np.mean(np.logaddexp(0, z) - y * z)


def g(b):
    return X.T @ (1 / (1 + np.exp(-X @ b)) - y) / n


def H(b):
    pr = 1 / (1 + np.exp(-X @ b))
    return (X * (pr * (1 - pr))[:, None]).T @ X / n


problems = {"Rosenbrock d=10": (rosen, rosen_der, rosen_hess, np.zeros(10)),
            "logit n=10000, p=20": (f, g, H, np.zeros(p))}
methods = ["Nelder-Mead", "CG", "BFGS", "L-BFGS-B", "Newton-CG", "trust-exact"]
for pname, (fun, jac, hess, x0) in problems.items():
    print(pname)
    print(f"  {'method':<12}{'nit':>6}{'nfev':>7}{'njev':>6}{'f*':>13}"
          f"{'ms':>8}")
    for m in methods:
        kw = {}
        if m != "Nelder-Mead":
            kw["jac"] = jac
        if m in ("Newton-CG", "trust-exact"):
            kw["hess"] = hess
        opts = {"maxiter": 3000, "maxfev": 3000} if m == "Nelder-Mead" \
            else {}
        t0 = time.perf_counter()
        res = minimize(fun, x0, method=m, options=opts, **kw)
        ms = 1000 * (time.perf_counter() - t0)
        nj = getattr(res, "njev", 0)
        print(f"  {m:<12}{res.nit:>6}{res.nfev:>7}{nj:>6}{res.fun:>13.6e}"
              f"{ms:>8.0f}")
