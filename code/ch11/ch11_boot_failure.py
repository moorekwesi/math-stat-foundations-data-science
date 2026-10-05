# DEP 803 -- Chapter 11: Resampling: Bootstrap, Permutation Tests, Intervals
# When the bootstrap fails: the sample maximum and an infinite-variance mean.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)

# (a) theta_hat = max(X), X ~ Uniform(0, theta); true law of n(theta-max)/theta
theta, n, B = 1.0, 50, 10000
x = rng.uniform(0, theta, size=n)
t_hat = x.max()
t_np = x[rng.integers(0, n, size=(B, n))].max(axis=1)        # nonparametric
t_par = rng.uniform(0, t_hat, size=(B, n)).max(axis=1)      # parametric
z_np = n * (t_hat - t_np) / t_hat
z_par = n * (t_hat - t_par) / t_hat
print("(a) maximum of a Uniform(0, theta) sample, n = 50")
print(f"  P*(theta* = theta_hat): nonparametric = {np.mean(t_np == t_hat):.3f}"
      f", theory 1-(1-1/n)^n = {1 - (1 - 1 / n)**n:.3f}")
print(f"  P*(theta* = theta_hat): parametric    = {np.mean(t_par == t_hat):.3f}"
      f" (true limit law is continuous)")

fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.4), sharey=True)
grid = np.linspace(0, 6, 200)
for ax, z, ttl in zip(axes, (z_np, z_par), ("nonparametric", "parametric")):
    ax.hist(z, bins=np.arange(0, 6.25, 0.25), density=True,
            color="steelblue", alpha=0.75)
    ax.plot(grid, np.exp(-grid), "k--", lw=1.2, label="Exp(1) limit")
    ax.set_title(f"{ttl} bootstrap", fontsize=10)
    ax.set_xlabel(r"$n(\hat\theta-\hat\theta^*)/\hat\theta$")
axes[0].set_ylabel("density")
axes[0].legend(frameon=False)
plt.savefig("ch11_boot_max.pdf", bbox_inches="tight")

# (b) Coverage of the 95% percentile interval for a mean: light vs heavy tails
R, n, B = 1000, 100, 999
print("(b) coverage of the 95% percentile bootstrap CI for a mean, n = 100")
for label, alpha in (("Pareto, alpha = 4.0 (finite variance)", 4.0),
                     ("Pareto, alpha = 1.5 (infinite variance)", 1.5)):
    true_mean = alpha / (alpha - 1)
    hits = 0
    for r in range(R):
        s = rng.pareto(alpha, size=n) + 1.0     # Pareto with x_min = 1
        m = s[rng.integers(0, n, size=(B, n))].mean(axis=1)
        lo, hi = np.quantile(m, [0.025, 0.975])
        hits += lo <= true_mean <= hi
    print(f"  {label:<40} coverage = {hits / R:.3f}")
