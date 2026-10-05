# DEP 803 -- Chapter 5: Common Distributions in Data Science and Econometrics
# Pareto tails of top incomes: log-log survival plot and the Hill estimator
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
n = 20_000
logn = rng.lognormal(np.log(12_000), 0.8, n)          # log-normal body
alpha, xm = 2.0, 60_000.0                             # Pareto top tail
par = xm * (1 + rng.pareto(alpha, n))                 # classical Pareto
# mixture: 95% log-normal, 5% Pareto top earners (simulated)
income = np.where(rng.random(n) < 0.95, logn, par)


def hill(x, k):
    """Hill estimator of the tail index from the k largest observations."""
    xs = np.sort(x)[::-1]
    return 1.0 / np.mean(np.log(xs[:k] / xs[k]))


for k in [100, 200, 500, 1000]:
    print(f"Hill estimate of alpha with k = {k:4d}: {hill(income, k):.3f}")
print(f"true Pareto tail index alpha = {alpha}")
print(f"Pareto mean x_m*alpha/(alpha-1) = {xm * alpha / (alpha - 1):.0f}")
top1 = np.sort(income)[-n // 100:].sum() / income.sum()
print(f"top 1% income share: mixture = {top1:.4f}, "
      f"log-normal only = {np.sort(logn)[-n // 100:].sum() / logn.sum():.4f}")

fig, ax = plt.subplots(figsize=(6, 3.8))
for x, lab, col in [(logn, "log-normal only", "#888888"),
                    (income, "log-normal + Pareto tail", "#1f4e79")]:
    xs = np.sort(x)
    surv = 1 - np.arange(len(xs)) / len(xs)
    ax.loglog(xs, surv, lw=2, color=col, label=lab)
grid = np.logspace(np.log10(xm), 6.5, 50)
ax.loglog(grid, 0.05 * (grid / xm) ** (-alpha), "--", color="#cc9900",
          label="reference line, slope = -alpha")
ax.set_ylim(1e-4, 1.1)
ax.set_xlabel("income (simulated units)")
ax.set_ylabel("P(Income > x)")
ax.legend(frameon=False, fontsize=8)
plt.savefig("ch05_pareto_tail.pdf", bbox_inches="tight")
