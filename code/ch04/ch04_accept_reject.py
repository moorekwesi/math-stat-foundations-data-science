# DEP 803 -- Chapter 4: Probability Foundations, Random Variables and Transformations
# Acceptance-rejection sampling from a Beta(2.7, 6.3) density.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)
target = stats.beta(2.7, 6.3)                # e.g. a share of budget on food
mode = (2.7 - 1) / (2.7 + 6.3 - 2)
M = target.pdf(mode)                         # envelope M * g with g = U(0,1)
n = 200_000
X = rng.random(n)                            # proposals from g
V = rng.random(n)
accept = V * M <= target.pdf(X)              # accept if V <= f(X) / (M g(X))
draws = X[accept]

print(f"envelope constant M = sup f/g      = {M:.4f}")
print(f"acceptance rate: theory 1/M        = {1 / M:.4f}")
print(f"                 simulated         = {accept.mean():.4f}")
print(f"accepted draws                     = {draws.size}")
print(f"mean: target {target.mean():.4f}, sample {draws.mean():.4f}")
print(f"var : target {target.var():.5f}, sample {draws.var():.5f}")
print(f"KS p-value (first 20000 draws)     = "
      f"{stats.kstest(draws[:20_000], target.cdf).pvalue:.3f}")

fig, ax = plt.subplots(figsize=(6, 3.8))
k = 3000
ax.scatter(X[:k][accept[:k]], (V * M)[:k][accept[:k]], s=3, color="C2",
           label="accepted")
ax.scatter(X[:k][~accept[:k]], (V * M)[:k][~accept[:k]], s=3, color="C3",
           alpha=0.4, label="rejected")
g = np.linspace(0, 1, 300)
ax.plot(g, target.pdf(g), "k", lw=2, label="target f")
ax.axhline(M, color="k", ls="--", lw=1, label="envelope M g")
ax.set(xlabel="x", ylabel="height V M g(X)")
ax.legend(frameon=False, loc="upper right")
ax.grid(alpha=0.3)
plt.savefig("ch04_accept_reject.pdf", bbox_inches="tight")
