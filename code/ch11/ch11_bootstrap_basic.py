# DEP 803 -- Chapter 11: Resampling: Bootstrap, Permutation Tests, Intervals
# Nonparametric bootstrap standard errors and bias for four statistics.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)


def gini(x, axis=-1):
    """Plug-in Gini coefficient along the last axis (`axis` is for scipy)."""
    x = np.sort(x, axis=-1)
    n = x.shape[-1]
    w = 2 * np.arange(1, n + 1) - n - 1
    return np.sum(w * x, axis=-1) / (n * np.sum(x, axis=-1))


def draw(size):
    """Simulated household survey: income and food expenditure."""
    inc = rng.lognormal(7.0, 0.9, size=size)
    food = 0.45 * inc**0.85 * rng.lognormal(0, 0.3, size=size)
    return inc, food


stats_fns = {
    "mean income": lambda inc, food: inc.mean(axis=-1),
    "median income": lambda inc, food: np.median(inc, axis=-1),
    "Gini": lambda inc, food: gini(inc),
    "food share (ratio)": lambda inc, food: food.mean(-1) / inc.mean(-1),
}

n, B = 150, 4000
inc, food = draw(n)                          # the one observed sample
idx = rng.integers(0, n, size=(B, n))        # B resamples of row indices
inc_b, food_b = inc[idx], food[idx]

# "Truth" for comparison: sampling distribution from 20000 fresh surveys
inc_mc, food_mc = draw((20000, n))

print(f"n = {n}, B = {B} bootstrap replications")
print(f"{'statistic':<20}{'estimate':>10}{'boot se':>10}{'true se':>10}"
      f"{'boot bias':>11}")
for name, f in stats_fns.items():
    t_hat = f(inc, food)
    t_star = f(inc_b, food_b)
    se_boot = t_star.std(ddof=1)
    bias_boot = t_star.mean() - t_hat
    se_true = f(inc_mc, food_mc).std(ddof=1)
    print(f"{name:<20}{t_hat:10.4f}{se_boot:10.4f}{se_true:10.4f}"
          f"{bias_boot:11.4f}")
print(f"analytic se of mean, s/sqrt(n) = {inc.std(ddof=1) / np.sqrt(n):.4f}")

g_hat, g_star = gini(inc), gini(inc_b)
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.hist(g_star, bins=50, color="steelblue", alpha=0.75, density=True,
        label="bootstrap replicates $G^*$")
ax.axvline(g_hat, color="black", lw=1.2, label=r"sample Gini $\hat G$")
ax.axvline(g_star.mean(), color="firebrick", lw=1.2, ls="--",
           label="mean of $G^*$")
ax.set_xlabel("Gini coefficient")
ax.set_ylabel("density")
ax.legend(frameon=False)
plt.savefig("ch11_boot_hist.pdf", bbox_inches="tight")
