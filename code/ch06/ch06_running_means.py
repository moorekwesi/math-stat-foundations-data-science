# DEP 803 -- Chapter 6: Expectation, Covariance, Limit Theorems and Concentration
# Running sample means: the LLN at work, and its failure for the Cauchy law.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
N = 100_000
n = np.arange(1, N + 1)
alpha = 1.5                                        # Pareto tail index
draws = {
    "Bernoulli(0.3)": (rng.random(N) < 0.3).astype(float),
    "Exponential(1)": rng.exponential(1.0, N),
    "Pareto(1.5)": rng.pareto(alpha, N) + 1.0,      # mean 3, infinite variance
    "Cauchy": rng.standard_cauchy(N),
}
means = {"Bernoulli(0.3)": 0.3, "Exponential(1)": 1.0,
         "Pareto(1.5)": alpha / (alpha - 1), "Cauchy": np.nan}

print(f"{'law':>15}{'mu':>7}{'n=100':>9}{'n=1e3':>9}{'n=1e4':>9}{'n=1e5':>9}")
run = {}
for lab, x in draws.items():
    run[lab] = np.cumsum(x) / n
    vals = "".join(f"{run[lab][m - 1]:9.3f}" for m in [100, 1000, 10000, N])
    print(f"{lab:>15}{means[lab]:7.2f}{vals}")

plt.rcParams.update({"font.size": 10})
fig, axes = plt.subplots(2, 2, figsize=(6.4, 4.4), sharex=True)
for ax, lab in zip(axes.ravel(), draws):
    ax.plot(n, run[lab], lw=1, color="navy")
    if not np.isnan(means[lab]):
        ax.axhline(means[lab], color="firebrick", ls="--", lw=1)
    ax.set_xscale("log")
    ax.set_title(lab, fontsize=10)
for ax in axes[1]:
    ax.set_xlabel("n (log scale)")
for ax in axes[:, 0]:
    ax.set_ylabel("running mean")
fig.tight_layout()
plt.savefig("ch06_lln.pdf", bbox_inches="tight")
