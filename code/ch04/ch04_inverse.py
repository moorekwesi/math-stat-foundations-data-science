# DEP 803 -- Chapter 4: Probability Foundations, Random Variables and Transformations
# Inverse transform sampling: exponential, Pareto and a discrete distribution.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
from scipy import stats

rng = np.random.default_rng(803)
n = 500_000
U = rng.random(n)

# Exponential with rate lam: F^{-1}(u) = -log(1 - u) / lam
lam = 0.25
E = -np.log1p(-U) / lam
print(f"Exponential(rate 0.25): mean {E.mean():.3f} (4.000), "
      f"var {E.var():.3f} (16.000)")

# Pareto top incomes: F(x) = 1 - (xm/x)^a for x >= xm, F^{-1}(u) = xm (1-u)^(-1/a)
xm, a = 50.0, 2.5
Pa = xm * (1 - U) ** (-1 / a)
print(f"Pareto(xm=50, a=2.5):   mean {Pa.mean():.2f} ({a * xm / (a - 1):.2f}), "
      f"P(X > 200) {np.mean(Pa > 200):.4f} ({(xm / 200) ** a:.4f})")
print(f"  KS p-value vs scipy pareto: "
      f"{stats.kstest(Pa[:20_000], 'pareto', (a, 0, xm)).pvalue:.3f}")

# Discrete: household size with an arbitrary PMF, via the generalised inverse
size = np.arange(1, 9)
pmf = np.array([0.10, 0.14, 0.17, 0.18, 0.15, 0.11, 0.08, 0.07])
cdf = np.cumsum(pmf)
idx = np.searchsorted(cdf, U, side="right").clip(max=size.size - 1)
draw = size[idx]
freq = np.bincount(draw, minlength=9)[1:] / n
print("\nHousehold size  " + " ".join(f"{s:>6d}" for s in size))
print("target PMF      " + " ".join(f"{p:6.3f}" for p in pmf))
print("sample freq.    " + " ".join(f"{f:6.3f}" for f in freq))
print(f"mean size: target {np.dot(size, pmf):.3f}, sample {draw.mean():.3f}")
