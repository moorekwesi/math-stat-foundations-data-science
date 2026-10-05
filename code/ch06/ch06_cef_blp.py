# DEP 803 -- Chapter 6: Expectation, Covariance, Limit Theorems and Concentration
# The CEF minimises mean-squared error; the linear projection is its best line.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
# Simulated: X = years of schooling (0-16), Y = log earnings, nonlinear CEF
n = 50_000
x = rng.uniform(0, 16, n)
cef = lambda t: 1.0 + 0.02 * t + 0.006 * t**2      # convex returns
y = cef(x) + rng.normal(0, 0.3 + 0.02 * x)         # heteroskedastic noise

# Linear projection from moments: b = Cov(X,Y)/Var(X), a = E[Y] - b E[X]
b = np.cov(x, y)[0, 1] / np.var(x, ddof=1)
a = y.mean() - b * x.mean()
# Nonparametric CEF estimate: means within 16 bins
edges = np.linspace(0, 16, 17)
idx = np.clip(np.digitize(x, edges) - 1, 0, 15)
bin_mean = np.bincount(idx, weights=y) / np.bincount(idx)

mse = lambda pred: np.mean((y - pred)**2)
print(f"linear projection: a = {a:.4f}, b = {b:.4f}")
print(f"MSE of true CEF          : {mse(cef(x)):.5f}")
print(f"MSE of binned CEF        : {mse(bin_mean[idx]):.5f}")
print(f"MSE of linear projection : {mse(a + b * x):.5f}")
print(f"MSE of constant E[Y]     : {mse(y.mean()):.5f}")
# Orthogonality: the CEF error is uncorrelated with ANY function of X
corr = lambda a, b: np.corrcoef(a, b)[0, 1]
e = y - cef(x)
print(f"CEF error:  corr(e, X) = {corr(e, x): .4f}  corr(e, X^2) = "
      f"{corr(e, x**2): .4f}  corr(e, sin X) = {corr(e, np.sin(x)): .4f}")
u = y - a - b * x
print(f"projection: corr(u, X) = {corr(u, x): .4f}  corr(u, X^2) = "
      f"{corr(u, x**2): .4f}")

plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(figsize=(6, 3.8))
sub = rng.choice(n, 1500, replace=False)
ax.scatter(x[sub], y[sub], s=4, alpha=0.3, color="gray", label="data (subsample)")
t = np.linspace(0, 16, 200)
ax.plot(t, cef(t), color="navy", lw=2, label="CEF E[Y|X]")
ax.plot(t, a + b * t, color="firebrick", lw=2, ls="--", label="linear projection")
mid = (edges[:-1] + edges[1:]) / 2
ax.plot(mid, bin_mean, "o", color="darkgreen", ms=4, label="binned means")
ax.set_xlabel("years of schooling X (simulated)")
ax.set_ylabel("log earnings Y")
ax.legend(frameon=False, fontsize=9)
plt.savefig("ch06_cef.pdf", bbox_inches="tight")
