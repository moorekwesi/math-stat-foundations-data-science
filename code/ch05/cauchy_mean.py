# DEP 803 -- Chapter 5: Common Distributions in Data Science and Econometrics
# The Cauchy warning: running sample means that never settle down
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
n = 100_000
idx = np.arange(1, n + 1)
normal = rng.standard_normal(n)
cauchy = rng.standard_normal(n) / rng.standard_normal(n)   # ratio of normals

run_norm = np.cumsum(normal) / idx
run_cauchy = np.cumsum(cauchy) / idx
print("    n    mean(normal)   mean(Cauchy)   median(Cauchy)")
for m in [10, 100, 1000, 10_000, 100_000]:
    print(f"{m:6d}{run_norm[m - 1]:14.4f}{run_cauchy[m - 1]:15.4f}"
          f"{np.median(cauchy[:m]):16.4f}")

# the mean of n Cauchy draws is again standard Cauchy: check its IQR
means = (rng.standard_normal((20_000, 50))
         / rng.standard_normal((20_000, 50))).mean(axis=1)
q1, q3 = np.quantile(means, [0.25, 0.75])
print(f"IQR of means of 50 Cauchy draws = [{q1:.3f}, {q3:.3f}] (theory [-1, 1])")

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.semilogx(idx, run_cauchy, color="#b22222", lw=1.2, label="Cauchy")
ax.semilogx(idx, run_norm, color="#1f4e79", lw=1.8, label="normal")
ax.axhline(0, color="k", lw=0.6)
ax.set_xlabel("sample size n")
ax.set_ylabel("running sample mean")
ax.legend(frameon=False)
plt.savefig("ch05_cauchy_mean.pdf", bbox_inches="tight")
