# DEP 803 -- Chapter 9: Constrained Optimisation and Lagrange Multipliers
# KKT conditions: hand-solved examples checked numerically with SLSQP.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize


def kkt_multipliers(grad_f, grads_active):
    """Solve grad f + sum mu_j grad g_j = 0 for mu (least squares)."""
    G = np.column_stack(grads_active)
    mu, *_ = np.linalg.lstsq(G, -grad_f, rcond=None)
    return mu, np.linalg.norm(G @ mu + grad_f)


# ---- Example A: min (x1-2)^2 + (x2-2)^2, x1+x2 <= 2, x1 <= 1.5, x >= 0 -----
f = lambda x: (x[0] - 2)**2 + (x[1] - 2)**2
df = lambda x: np.array([2 * (x[0] - 2), 2 * (x[1] - 2)])
cons = [{"type": "ineq", "fun": lambda x: 2 - x[0] - x[1]},     # g1 <= 0
        {"type": "ineq", "fun": lambda x: 1.5 - x[0]}]          # g2 <= 0
res = minimize(f, x0=[0.0, 0.0], jac=df, method="SLSQP", constraints=cons,
               bounds=[(0, None), (0, None)], tol=1e-12)
x = res.x
slack = np.array([2 - x.sum(), 1.5 - x[0], x[0], x[1]])
print("Example A (by hand: x* = (1, 1), mu1 = 2, others 0)")
print("  SLSQP x* =", np.round(x, 6), " f* = %.6f" % res.fun)
print("  slacks of (g1, g2, x1>=0, x2>=0):", np.round(slack, 6))
mu, resid = kkt_multipliers(df(x), [np.array([1.0, 1.0])])
print("  multiplier on active g1: mu1 = %.6f  (stationarity residual %.1e)"
      % (mu[0], resid))

# ---- Example B: quasi-linear utility, corner solution --------------------
# max alpha*log(x1) + x2  s.t.  p1 x1 + p2 x2 <= m,  x1, x2 >= 0
alpha, p1, p2 = 3.0, 1.0, 2.0


def demand_hand(m):
    if m > alpha * p2:                                   # interior
        return np.array([alpha * p2 / p1, (m - alpha * p2) / p2]), 1 / p2
    return np.array([m / p1, 0.0]), alpha / m            # corner: x2 = 0


print("\nExample B: quasi-linear utility, threshold income alpha*p2 = %.1f"
      % (alpha * p2))
print("     m   x1(hand) x2(hand)  x1(SLSQP) x2(SLSQP)  lambda   mu2(x2>=0)")
for m in [2.0, 4.0, 6.0, 8.0, 12.0]:
    xh, lam = demand_hand(m)
    r = minimize(lambda z: -(alpha * np.log(z[0]) + z[1]), x0=[0.5, 0.5],
                 method="SLSQP", bounds=[(1e-9, None), (0, None)],
                 constraints=[{"type": "ineq",
                               "fun": lambda z: m - p1 * z[0] - p2 * z[1]}],
                 tol=1e-12)
    mu2 = max(lam * p2 - 1, 0.0)                         # dual feasibility
    print("  %5.1f  %8.3f %8.3f  %9.3f %9.3f  %7.4f  %8.4f" %
          (m, *xh, *r.x, lam, mu2))

# ---- Example C: failure of LICQ (Kuhn-Tucker cusp) ------------------------
# min -x1  s.t.  x2 - (1 - x1)^3 <= 0,  -x2 <= 0 ; optimum at (1, 0)
xc = np.array([1.0, 0.0])
g_act = [np.array([3 * (1 - xc[0])**2, 1.0]), np.array([0.0, -1.0])]
mu, resid = kkt_multipliers(np.array([-1.0, 0.0]), g_act)
print("\nExample C (cusp): best multipliers", np.round(mu, 3),
      " stationarity residual = %.3f -> no KKT point" % resid)

# ---- figure ---------------------------------------------------------------
fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
ax = axs[0]
g1, g2 = np.meshgrid(np.linspace(-0.2, 2.6, 300), np.linspace(-0.2, 2.6, 300))
ax.contour(g1, g2, (g1 - 2)**2 + (g2 - 2)**2, levels=[0.5, 2, 4.5],
           colors="grey", linewidths=0.8)
ax.fill([0, 1.5, 1.5, 0], [0, 0, 0.5, 2], color="C0", alpha=0.2,
        label="feasible set")
ax.plot(2, 2, "kx")
ax.plot(1, 1, "o", color="C3", label="x* = (1, 1)")
ax.arrow(1, 1, 0.35, 0.35, color="C3", width=0.01, head_width=0.06)
ax.annotate("-grad f = mu1 grad g1", (1.05, 1.5), color="C3", fontsize=8)
ax.set_xlabel("$x_1$")
ax.set_ylabel("$x_2$")
ax.set_title("Example A: one active constraint", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
ax = axs[1]
ms = np.linspace(0.2, 14, 200)
X = np.array([demand_hand(mm)[0] for mm in ms])
ax.plot(ms, X[:, 0], lw=2, label="$x_1(m)$")
ax.plot(ms, X[:, 1], lw=2, label="$x_2(m)$")
ax.axvline(alpha * p2, ls="--", color="grey")
ax.annotate("corner   interior", (alpha * p2, 2.0), fontsize=8, ha="center")
ax.set_xlabel("income m")
ax.set_ylabel("demand")
ax.set_title("Example B: Engel curves with a kink", fontsize=10)
ax.legend(fontsize=8)
for a_ in axs:
    a_.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ch09_kkt_examples.pdf", bbox_inches="tight")
