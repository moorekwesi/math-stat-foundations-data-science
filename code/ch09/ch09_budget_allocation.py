# DEP 803 -- Chapter 9: Constrained Optimisation and Lagrange Multipliers
# District budget allocation across programmes: KKT "water filling".
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq, minimize

# Simulated returns: outcome_i(x) = a_i log(1 + x_i / b_i), x in GHS million
names = ["education", "health", "agric. ext.", "water/sanit.", "feeder roads"]
a = np.array([3.0, 4.0, 2.0, 2.5, 1.5])
b = np.array([1.0, 2.0, 0.5, 1.5, 0.6])
first = a / b                                  # marginal return at x_i = 0


def allocate(B):
    """KKT: x_i = max(0, a_i/lam - b_i) with sum x_i = B (bisection)."""
    lam = brentq(lambda L: np.maximum(a / L - b, 0).sum() - B, 1e-9,
                 first.max(), xtol=1e-14)
    return np.maximum(a / lam - b, 0), lam


B0 = 1.8
x, lam = allocate(B0)
obj = lambda z: -np.sum(a * np.log1p(z / b))
res = minimize(obj, np.full(5, B0 / 5), method="SLSQP", tol=1e-12,
               bounds=[(0, None)] * 5,
               constraints=[{"type": "ineq", "fun": lambda z: B0 - z.sum()}])
print("Budget B = GHS %.1f million (simulated returns)" % B0)
print("  programme     a_i/b_i   x_i(KKT)  x_i(SLSQP)  marg. return")
for i in range(5):
    print("  %-13s %7.2f  %8.3f  %9.3f   %8.4f" %
          (names[i], first[i], x[i], res.x[i], a[i] / (b[i] + x[i])))
print("  lambda = %.4f: programmes funded iff a_i/b_i > lambda" % lam)
print("  outcome index V(B) = %.4f" % -obj(x))

# expansion path: who is funded as the budget grows
Bs = np.linspace(0.01, 15, 300)
path = np.array([allocate(B)[0] for B in Bs])
lams = np.array([allocate(B)[1] for B in Bs])
print("\nentry budgets (B at which each programme starts):")
for i in np.argsort(-first):
    Bi = np.maximum(a / first[i] - b, 0).sum()
    print("  %-13s B = %.3f" % (names[i], Bi))

fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
for i in range(5):
    axs[0].plot(Bs, path[:, i], lw=2, label=names[i])
axs[0].set_xlabel("budget B (GHS million)")
axs[0].set_ylabel("allocation x_i(B)")
axs[0].set_title("Expansion path (programmes enter in turn)", fontsize=10)
axs[0].legend(fontsize=7)
axs[1].plot(Bs, lams, lw=2, color="C3")
axs[1].set_xlabel("budget B (GHS million)")
axs[1].set_ylabel("shadow value lambda(B)")
axs[1].set_title("Marginal value of the budget", fontsize=10)
for ax in axs:
    ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("ch09_budget_allocation.pdf", bbox_inches="tight")
