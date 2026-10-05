# DEP 803 -- Chapter 4: Probability Foundations, Random Variables and Transformations
# Change of variables: log-normal incomes from normal log-incomes.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)
mu, sigma = 1.0, 0.8                         # parameters of log income X
X = rng.normal(mu, sigma, 200_000)
Y = np.exp(X)                                # income Y = g(X) = exp(X)


def f_Y(y):
    """f_Y(y) = f_X(log y) |d log y / dy| = f_X(log y) / y."""
    return stats.norm.pdf(np.log(y), mu, sigma) / y


print("Income Y = exp(X), X ~ N(1.0, 0.8^2)")
print(f"{'quantity':<12} {'formula':>9} {'simulated':>10}")
rows = [("mean", np.exp(mu + sigma**2 / 2), Y.mean()),
        ("median", np.exp(mu), np.median(Y)),
        ("variance", (np.exp(sigma**2) - 1) * np.exp(2 * mu + sigma**2), Y.var()),
        ("P(Y <= 2)", stats.norm.cdf((np.log(2) - mu) / sigma), np.mean(Y <= 2))]
for name, a, b in rows:
    print(f"{name:<12} {a:9.4f} {b:10.4f}")
print(f"mode exp(mu - sigma^2) = {np.exp(mu - sigma**2):.4f}"
      "  < median < mean (right skew)")
# The density formula integrates to one and agrees with scipy.stats.lognorm
g = np.linspace(1e-4, 80, 400_001)
print(f"integral of f_Y over (0, 80)        = {np.trapz(f_Y(g), g):.6f}")
ln = stats.lognorm(s=sigma, scale=np.exp(mu))
print(f"max |f_Y - scipy lognorm.pdf|       = {np.max(abs(f_Y(g) - ln.pdf(g))):.2e}")

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.4))
axes[0].hist(X, bins=80, density=True, alpha=0.5)
xg = np.linspace(-2.5, 4.5, 300)
axes[0].plot(xg, stats.norm.pdf(xg, mu, sigma), "k", lw=1.5)
axes[0].set_xlabel("log income X")
axes[0].set_title("X ~ normal")
axes[1].hist(Y[Y < 20], bins=80, density=True, alpha=0.5, color="C1",
             weights=np.full((Y < 20).sum(), np.mean(Y < 20)))
yg = np.linspace(0.01, 20, 400)
axes[1].plot(yg, f_Y(yg), "k", lw=1.5, label="f_Y from formula")
axes[1].set_xlabel("income Y = exp(X)")
axes[1].set_title("Y ~ log-normal")
axes[1].legend(frameon=False)
for a in axes:
    a.set_ylabel("density")
    a.grid(alpha=0.3)
fig.tight_layout()
plt.savefig("ch04_lognormal.pdf", bbox_inches="tight")
