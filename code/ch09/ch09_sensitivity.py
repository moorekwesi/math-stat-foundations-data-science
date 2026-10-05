# DEP 803 -- Chapter 9: Constrained Optimisation and Lagrange Multipliers
# Sensitivity of an optimal allocation: implicit function theorem + Monte Carlo.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import brentq

names = ["education", "health", "agric. ext.", "water/sanit.", "feeder roads"]
a = np.array([3.0, 4.0, 2.0, 2.5, 1.5])      # same simulated returns as
b = np.array([1.0, 2.0, 0.5, 1.5, 0.6])      # ch09_budget_allocation.py
B = 4.0


def allocate(a, B):
    lam = brentq(lambda L: np.maximum(a / L - b, 0).sum() - B, 1e-9,
                 (a / b).max(), xtol=1e-14)
    return np.maximum(a / lam - b, 0), lam


x, lam = allocate(a, B)
# KKT system F(x, lam; a, B) = [a/(b+x) - lam ; sum x - B] = 0 (all active)
n = len(a)
J = np.block([[np.diag(-a / (b + x)**2), -np.ones((n, 1))],
              [np.ones((1, n)), np.zeros((1, 1))]])
dF_da = np.vstack([np.diag(1 / (b + x)), np.zeros((1, n))])
dF_dB = np.r_[np.zeros(n), -1.0]
S = -np.linalg.solve(J, np.column_stack([dF_da, dF_dB]))  # d(x,lam)/d(a,B)
h = 1e-6
fd = (allocate(a + h * np.eye(n)[1], B)[0] -
      allocate(a - h * np.eye(n)[1], B)[0]) / (2 * h)
print("Allocation at B = %.1f:" % B, np.round(x, 3), " lambda = %.4f" % lam)
print("dx/da_health (IFT):        ", np.round(S[:n, 1], 4))
print("dx/da_health (finite diff):", np.round(fd, 4))
print("dx/dB (IFT):               ", np.round(S[:n, n], 4),
      " dlam/dB = %.4f" % S[n, n])
el = S[:n, 1] * a[1] / np.where(x > 0, x, np.nan)
print("elasticity of x_i w.r.t. a_health:", np.round(el, 3))

# Monte Carlo: returns uncertain, a_i * exp(N(0, 0.25^2))
rng = np.random.default_rng(803)
R = 2000
draws = a * np.exp(rng.normal(0, 0.25, (R, n)))
X = np.array([allocate(d, B)[0] for d in draws])
loss = np.array([np.sum(d * np.log1p(allocate(d, B)[0] / b)) -
                 np.sum(d * np.log1p(x / b)) for d in draws])
print("\nMonte Carlo (R = %d, 25%% lognormal uncertainty in returns):" % R)
print("  programme      x_nominal  mean   5%   95%   P(funded)")
for i in range(n):
    q5, q95 = np.percentile(X[:, i], [5, 95])
    print("  %-13s %8.3f  %5.2f %5.2f %5.2f   %5.2f" %
          (names[i], x[i], X[:, i].mean(), q5, q95, np.mean(X[:, i] > 0)))
print("  regret of nominal plan: mean %.4f, 95th pct %.4f (outcome units)" %
      (loss.mean(), np.percentile(loss, 95)))

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.boxplot([X[:, i] for i in range(n)], labels=names, showfliers=False)
ax.plot(range(1, n + 1), x, "D", color="C3", label="nominal optimum")
ax.set_ylabel("allocation (GHS million)")
ax.set_title("Sensitivity of the optimal allocation to uncertain returns",
             fontsize=10)
ax.tick_params(axis="x", labelsize=8)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.savefig("ch09_sensitivity.pdf", bbox_inches="tight")
