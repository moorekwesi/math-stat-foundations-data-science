# DEP 803 -- Chapter 4: Probability Foundations, Random Variables and Transformations
# Diagnostic test: positive and negative predictive values versus prevalence.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
sens, spec = 0.95, 0.95                     # illustrative test characteristics


def ppv(p):
    return sens * p / (sens * p + (1 - spec) * (1 - p))


def npv(p):
    return spec * (1 - p) / (spec * (1 - p) + (1 - sens) * p)


print(f"Rapid test with sensitivity {sens} and specificity {spec}")
print(f"{'prevalence':>10} {'PPV':>7} {'NPV':>7} {'PPV sim':>8}")
N = 1_000_000
for p in [0.01, 0.05, 0.10, 0.30, 0.50]:
    ill = rng.random(N) < p
    pos = np.where(ill, rng.random(N) < sens, rng.random(N) < 1 - spec)
    print(f"{p:10.2f} {ppv(p):7.3f} {npv(p):7.3f} {ill[pos].mean():8.3f}")

# Two positive tests, conditionally independent given true status
p = 0.05
post1 = ppv(p)
post2 = sens * post1 / (sens * post1 + (1 - spec) * (1 - post1))
print(f"\nPrevalence 0.05: P(ill | +) = {post1:.3f}, P(ill | +,+) = {post2:.3f}")

grid = np.linspace(0.001, 0.999, 400)
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(grid, ppv(grid), label="PPV = P(ill | positive)")
ax.plot(grid, npv(grid), label="NPV = P(healthy | negative)")
ax.set_xscale("logit")
ax.set(xlabel="prevalence P(ill) (logit scale)", ylabel="predictive value")
ax.legend(frameon=False, loc="lower right")
ax.grid(alpha=0.3)
plt.savefig("ch04_malaria.pdf", bbox_inches="tight")
