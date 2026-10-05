# DEP 803 -- Chapter 10: Numerical Integration and Monte Carlo Simulation
# Inverse-CDF sampling and acceptance-rejection sampling.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)
n = 100_000
u = rng.random(n)

# Inverse CDF: Pareto top incomes, F(x) = 1 - (xm/x)^alpha  ->  x = xm u^(-1/a)
xm, alpha = 5000.0, 2.5
x = xm * (1 - u) ** (-1 / alpha)
ks = stats.kstest(x, stats.pareto(alpha, scale=xm).cdf)
print(f"Pareto by inverse CDF: mean {x.mean():.1f} (theory "
      f"{alpha * xm / (alpha - 1):.1f}), KS p-value {ks.pvalue:.3f}")

# Inverse CDF for a discrete distribution: household size 1..6
probs = np.array([0.10, 0.18, 0.22, 0.20, 0.17, 0.13])
size = 1 + np.searchsorted(np.cumsum(probs), u)
print("household size freq :", np.round(np.bincount(size)[1:] / n, 3))
print("target probabilities:", probs)

# Acceptance-rejection: Beta(2.7, 6.3) with a U(0,1) envelope, M = max density
a, b = 2.7, 6.3
target = stats.beta(a, b)
mode = (a - 1) / (a + b - 2)
M = target.pdf(mode)
prop = rng.random(n)
acc = rng.random(n) <= target.pdf(prop) / M
draws = prop[acc]
print(f"\nAR for Beta({a},{b}): M = {M:.4f}, acceptance rate "
      f"{acc.mean():.4f} (theory 1/M = {1 / M:.4f})")
print(f"   sample mean {draws.mean():.4f} (theory {a / (a + b):.4f}), "
      f"KS p-value {stats.kstest(draws, target.cdf).pvalue:.3f}")

# AR for a truncated normal: wages above a minimum wage c (standardised)
for c in (0.0, 1.0, 2.0, 3.0):
    z = rng.standard_normal(n)
    print(f"   N(0,1) truncated at c = {c:.0f}: naive acceptance "
          f"{np.mean(z > c):.4f}")

fig, ax = plt.subplots(figsize=(6, 3.8))
grid = np.linspace(0, 1, 300)
ax.hist(draws, bins=60, density=True, color="steelblue", alpha=0.7,
        label=f"accepted draws ({acc.sum()})")
ax.plot(grid, target.pdf(grid), color="navy", lw=2, label="target Beta(2.7,6.3)")
ax.axhline(M, color="darkorange", ls="--", label="envelope M x U(0,1) density")
ax.set_xlabel("x")
ax.set_ylabel("density")
ax.legend()
plt.savefig("ch10_accept_reject.pdf", bbox_inches="tight")
