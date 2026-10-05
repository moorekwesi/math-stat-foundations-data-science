# DEP 803 -- Chapter 4: Probability Foundations, Random Variables and Transformations
# Simulated bivariate distribution of log income and log consumption.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
mu = np.array([7.0, 6.8])                    # means of (log income, log cons.)
s1, s2, rho = 0.70, 0.50, 0.80
Sigma = np.array([[s1**2, rho * s1 * s2],
                  [rho * s1 * s2, s2**2]])
n = 5_000
Y = rng.multivariate_normal(mu, Sigma, size=n)
y1, y2 = Y[:, 0], Y[:, 1]

print("Sample moments of (log income, log consumption)")
print(f"  means       {y1.mean():.3f}  {y2.mean():.3f}   (true 7.000  6.800)")
print(f"  covariance  {np.cov(y1, y2)[0, 1]:.4f}           "
      f"(true {Sigma[0, 1]:.4f})")
print(f"  correlation {np.corrcoef(y1, y2)[0, 1]:.4f}           (true {rho:.4f})")
inc, con = np.exp(y1), np.exp(y2)
print(f"  corr. of levels exp(Y1), exp(Y2): {np.corrcoef(inc, con)[0, 1]:.4f}")

# Conditional distribution of Y2 given Y1 = y: normal, linear mean
slope = rho * s2 / s1
csd = s2 * np.sqrt(1 - rho**2)
print(f"\nE[Y2 | Y1 = y] = {mu[1]:.2f} + {slope:.4f} (y - {mu[0]:.2f}),"
      f"  sd(Y2 | Y1) = {csd:.4f}")
print(f"{'income band':>14} {'mean Y2 (sim)':>14} {'theory':>8} {'sd (sim)':>9}")
for lo in [6.0, 6.5, 7.0, 7.5]:
    m = (y1 >= lo) & (y1 < lo + 0.1)
    centre = y1[m].mean()
    print(f"[{lo:.1f}, {lo + 0.1:.1f}) {y2[m].mean():14.3f} "
          f"{mu[1] + slope * (centre - mu[0]):8.3f} {y2[m].std():9.3f}")

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.scatter(y1, y2, s=3, alpha=0.25, label="simulated households")
g = np.linspace(4.8, 9.2, 50)
ax.plot(g, mu[1] + slope * (g - mu[0]), "k", lw=2, label="E[Y2 | Y1 = y]")
ax.plot(g, mu[1] + (g - mu[0]), "k:", lw=1, label="45-degree slope")
ax.set(xlabel="log income Y1", ylabel="log consumption Y2")
ax.legend(frameon=False, loc="upper left")
ax.grid(alpha=0.3)
plt.savefig("ch04_bivariate.pdf", bbox_inches="tight")
