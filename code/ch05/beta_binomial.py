# DEP 803 -- Chapter 5: Common Distributions in Data Science and Econometrics
# Bayesian preview: beta prior x binomial likelihood = beta posterior
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

# simulated pilot survey: 14 of 40 sampled households take up a cash transfer
n, y = 40, 14
a0, b0 = 4.0, 12.0                 # prior: take-up around 25% (prior mean 0.25)
grid = np.linspace(0, 1, 2001)
prior = stats.beta(a0, b0).pdf(grid)
lik = stats.binom.pmf(y, n, grid)
post_grid = prior * lik
post_grid /= np.trapz(post_grid, grid)       # normalise numerically
post = stats.beta(a0 + y, b0 + n - y)         # conjugate update

print(f"MLE y/n = {y / n:.4f}")
print(f"prior mean = {a0 / (a0 + b0):.4f}, posterior mean = {post.mean():.4f}")
w = (a0 + b0) / (a0 + b0 + n)
print(f"posterior mean = w*prior mean + (1-w)*MLE with w = {w:.4f}")
print(f"max |grid posterior - Beta({a0 + y:.0f},{b0 + n - y:.0f}) pdf| = "
      f"{np.abs(post_grid - post.pdf(grid)).max():.2e}")
lo, hi = post.ppf([0.025, 0.975])
print(f"95% credible interval: [{lo:.4f}, {hi:.4f}]")
print(f"P(take-up > 0.30 | data) = {post.sf(0.30):.4f}")

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(grid, prior, color="#888888", lw=2, label="prior Beta(4,12)")
ax.plot(grid, lik / np.trapz(lik, grid), color="#1f4e79", lw=2, ls="--",
        label="likelihood (rescaled)")
ax.plot(grid, post.pdf(grid), color="#cc9900", lw=2.5, label="posterior Beta(18,38)")
ax.set_xlim(0, 0.8)
ax.set_xlabel("take-up rate theta")
ax.set_ylabel("density")
ax.legend(frameon=False)
plt.savefig("ch05_beta_binomial.pdf", bbox_inches="tight")
