# DEP 803 -- Chapter 9: Constrained Optimisation and Lagrange Multipliers
# Projected gradient, quadratic penalty and log-barrier methods compared.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

rng = np.random.default_rng(803)
m, n = 30, 5
A = rng.normal(size=(m, n))
bvec = A @ np.array([0.6, -0.2, 0.5, 0.4, 0.1]) + rng.normal(0, 0.1, m)
B = 1.0                         # feasible set: x >= 0, sum(x) <= B
f = lambda x: 0.5 * np.sum((A @ x - bvec)**2)
df = lambda x: A.T @ (A @ x - bvec)
ref = minimize(f, np.full(n, 0.1), jac=df, method="SLSQP", tol=1e-14,
               bounds=[(0, None)] * n,
               constraints=[{"type": "ineq", "fun": lambda x: B - x.sum()}])
fstar = ref.fun


def project(v):
    """Euclidean projection onto {x >= 0, sum x <= B}."""
    w = np.maximum(v, 0)
    if w.sum() <= B:
        return w
    u = np.sort(v)[::-1]                      # simplex projection (sorting)
    css = np.cumsum(u) - B
    k = np.nonzero(u - css / np.arange(1, n + 1) > 0)[0][-1]
    return np.maximum(v - css[k] / (k + 1), 0)


L = np.linalg.eigvalsh(A.T @ A).max()        # Lipschitz constant of grad f
x = np.zeros(n)
gap_pg = []
for it in range(200):
    x = project(x - df(x) / L)
    gap_pg.append(f(x) - fstar)
print("reference (SLSQP) x* =", np.round(ref.x, 4), " f* = %.6f" % fstar)
print("projected gradient x =", np.round(x, 4), " gap after 200 it = %.1e"
      % gap_pg[-1])

# quadratic penalty: f + rho/2 [ sum(min(x,0)^2) + max(sum x - B, 0)^2 ]
print("\n   rho    penalty-method x                       violation")
for rho in [1e0, 1e1, 1e2, 1e3, 1e4]:
    pen = lambda x: f(x) + rho / 2 * (np.sum(np.minimum(x, 0)**2) +
                                      max(x.sum() - B, 0)**2)
    xp = minimize(pen, np.zeros(n), method="BFGS", tol=1e-12).x
    viol = max(xp.sum() - B, 0) + np.abs(np.minimum(xp, 0)).sum()
    print("  %6.0e  %s  %.2e" % (rho, np.round(xp, 4), viol))

# log barrier: t f(x) - sum log x_i - log(B - sum x), Newton with damping
H0 = A.T @ A
x = np.full(n, B / (n + 1))                   # strictly feasible start
print("\n      t     f(x_t) - f*    bound (n+1)/t")
gap_bar, ts = [], []
for t in [1, 10, 100, 1e3, 1e4, 1e5]:
    for _ in range(50):
        s = B - x.sum()
        g = t * df(x) - 1 / x + 1 / s
        H = t * H0 + np.diag(1 / x**2) + 1 / s**2
        d = -np.linalg.solve(H, g)
        step = 1.0
        while np.any(x + step * d <= 0) or (x + step * d).sum() >= B:
            step *= 0.5                       # stay strictly inside
        x = x + step * d
        if np.abs(g @ d) < 1e-12:
            break
    ts.append(t)
    gap_bar.append(f(x) - fstar)
    print("  %6.0e   %.3e      %.1e" % (t, gap_bar[-1], (n + 1) / t))

fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
axs[0].semilogy(np.maximum(gap_pg, 1e-16), lw=2)
axs[0].set_xlabel("iteration")
axs[0].set_ylabel("f(x_k) - f*")
axs[0].set_title("Projected gradient (step 1/L)", fontsize=10)
axs[1].loglog(ts, gap_bar, "o-", lw=2, label="barrier gap")
axs[1].loglog(ts, (n + 1) / np.array(ts), "--", label="(n+1)/t bound")
axs[1].set_xlabel("barrier parameter t")
axs[1].set_title("Log-barrier (interior-point) path", fontsize=10)
axs[1].legend(fontsize=8)
for ax in axs:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ch09_constrained_algorithms.pdf", bbox_inches="tight")
