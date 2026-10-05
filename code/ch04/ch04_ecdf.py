# DEP 803 -- Chapter 4: Probability Foundations, Random Variables and Transformations
# Empirical CDF of simulated incomes and the Glivenko-Cantelli theorem.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)
F = stats.lognorm(s=0.8, scale=np.exp(1.0))       # "true" income distribution


def ecdf(sample):
    xs = np.sort(sample)
    return xs, np.arange(1, xs.size + 1) / xs.size


def sup_distance(sample):
    """D_n = sup_x |F_n(x) - F(x)|, attained at the jump points."""
    xs, fn = ecdf(sample)
    Fx = F.cdf(xs)
    return max(np.max(fn - Fx), np.max(Fx - (fn - 1 / xs.size)))


print(f"{'n':>7} {'D_n':>8} {'sqrt(n) D_n':>12} {'DKW 95% band':>13}")
for n in [20, 200, 2_000, 20_000, 200_000]:
    d = sup_distance(F.rvs(size=n, random_state=rng))
    eps = np.sqrt(np.log(2 / 0.05) / (2 * n))      # Dvoretzky-Kiefer-Wolfowitz
    print(f"{n:>7} {d:8.4f} {np.sqrt(n) * d:12.3f} {eps:13.4f}")

x = F.rvs(size=200, random_state=rng)
print("\nQuantiles of a sample of n = 200 versus the true quantile function")
for p in [0.10, 0.25, 0.50, 0.75, 0.90]:
    print(f"  p = {p:.2f}: sample {np.quantile(x, p):6.2f}   true {F.ppf(p):6.2f}")

fig, ax = plt.subplots(figsize=(6, 3.8))
grid = np.linspace(0, 15, 400)
ax.plot(grid, F.cdf(grid), "k", lw=2, label="true CDF F")
for n, c in [(20, "C0"), (200, "C1")]:
    xs, fn = ecdf(F.rvs(size=n, random_state=rng))
    ax.step(np.r_[0, xs], np.r_[0, fn], where="post", color=c, lw=1,
            label=f"ECDF, n = {n}")
ax.set_xlim(0, 15)
ax.set(xlabel="income (simulated, thousands)", ylabel="cumulative probability")
ax.legend(frameon=False, loc="lower right")
ax.grid(alpha=0.3)
plt.savefig("ch04_ecdf.pdf", bbox_inches="tight")
