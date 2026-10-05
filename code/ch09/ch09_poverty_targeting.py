# DEP 803 -- Chapter 9: Constrained Optimisation and Lagrange Multipliers
# Poverty-gap-minimising regional transfers as a linear programme.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog
from scipy.sparse import csr_matrix, hstack, identity

rng = np.random.default_rng(803)
G, n = 6, 3000                               # regions, simulated households
region = rng.integers(0, G, n)
mu = np.log([180, 230, 300, 380, 470, 600])  # regional log-income means
y = np.exp(mu[region] + rng.normal(0, 0.55, n))   # GHS per adult per month
z = 300.0                                    # poverty line (simulated)


def pgap(inc):
    return np.mean(np.maximum(z - inc, 0)) / z      # FGT(1) index


def best_regional(Bud):
    """min sum_i s_i  s.t. s_i >= z - y_i - t_g(i), s >= 0, t >= 0,
    sum_i t_g(i) <= Bud.  Variables: (t_1..t_G, s_1..s_n)."""
    D = csr_matrix((np.ones(n), (np.arange(n), region)), shape=(n, G))
    A1 = hstack([-D, -identity(n)])          # -t_g - s_i <= y_i - z
    cnt = np.bincount(region, minlength=G).astype(float)
    A2 = csr_matrix(np.r_[cnt, np.zeros(n)][None, :])
    res = linprog(np.r_[np.zeros(G), np.ones(n)],
                  A_ub=hstack([A1.T, A2.T]).T.tocsr(),
                  b_ub=np.r_[y - z, Bud], bounds=[(0, None)] * (G + n),
                  method="highs")
    return res.x[:G], res, cnt


Bud = 60.0 * n                               # GHS 60 per household on average
t, res, cnt = best_regional(Bud)
print("Simulated survey: n = %d households, poverty line z = %.0f" % (n, z))
print("  headcount = %.3f   poverty gap index = %.4f" %
      (np.mean(y < z), pgap(y)))
print("  budget = GHS %.0f (GHS %.0f per household)" % (Bud, Bud / n))
print("\n  region  mean income  poor share  optimal transfer t_g")
for g in range(G):
    sel = region == g
    print("  %4d   %10.0f   %9.3f   %10.1f" %
          (g + 1, y[sel].mean(), np.mean(y[sel] < z), t[g]))
print("\n  poverty gap index: universal %.4f  regional LP %.4f" %
      (pgap(y + Bud / n), res.fun / n / z))
lam = -res.ineqlin.marginals[-1]
print("  shadow price of budget: total gap falls by %.3f per GHS 1" % lam)
print("  (perfect targeting would give 1.000 until the gap is closed)")

budgets = np.linspace(0, 250, 26) * n
uni = [pgap(y + Bb / n) for Bb in budgets]
reg = [best_regional(Bb)[1].fun / n / z for Bb in budgets]
perfect = np.maximum(pgap(y) - budgets / n / z, 0)
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(budgets / n, uni, lw=2, label="universal (equal) transfer")
ax.plot(budgets / n, reg, lw=2, label="optimal regional targeting (LP)")
ax.plot(budgets / n, perfect, "--", color="grey",
        label="perfect targeting (bound)")
ax.set_xlabel("budget per household (GHS per month)")
ax.set_ylabel("poverty gap index FGT(1)")
ax.set_title("Targeting transfers to minimise the poverty gap", fontsize=10)
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
plt.savefig("ch09_poverty_targeting.pdf", bbox_inches="tight")
