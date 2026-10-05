# DEP 803 -- Chapter 11: Resampling: Bootstrap, Permutation Tests, Intervals
# Pairs, residual and wild bootstrap for a heteroskedastic regression slope.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
n, B, R = 100, 499, 1000
b0, b1 = 1.0, 0.08            # log earnings on years of schooling (simulated)


def simulate(size):
    s = rng.uniform(0, 16, size=size)                   # years of schooling
    e = rng.normal(0, 0.02 * s**1.5, size=size)         # sd rises with s
    return s, b0 + b1 * s + e


def slope(x, y):
    """OLS slope along the last axis (works for stacked resamples)."""
    xc = x - x.mean(axis=-1, keepdims=True)
    return np.sum(xc * y, axis=-1) / np.sum(xc**2, axis=-1)


def all_se(x, y):
    """Classical, HC1, pairs, residual and wild bootstrap se of the slope."""
    b = slope(x, y)
    a = y.mean() - b * x.mean()
    u = y - a - b * x
    xc = x - x.mean()
    sxx = np.sum(xc**2)
    se_ols = np.sqrt(np.sum(u**2) / (n - 2) / sxx)
    se_hc1 = np.sqrt(n / (n - 2) * np.sum(xc**2 * u**2)) / sxx
    idx = rng.integers(0, n, size=(B, n))
    se_pairs = slope(x[idx], y[idx]).std(ddof=1)
    fit = a + b * x
    se_res = slope(x, fit + u[idx]).std(ddof=1)          # iid residual draws
    v = rng.choice([-1.0, 1.0], size=(B, n))             # Rademacher weights
    se_wild = slope(x, fit + u * v).std(ddof=1)
    return b, np.array([se_ols, se_hc1, se_pairs, se_res, se_wild])


names = ["classical OLS", "HC1 sandwich", "pairs bootstrap",
         "residual bootstrap", "wild (Rademacher)"]
x, y = simulate(n)
b, se = all_se(x, y)
print(f"one sample: slope = {b:.4f}")
xs, ys = simulate((20000, n))
se_true = slope(xs, ys).std(ddof=1)
print(f"Monte Carlo (true) sd of the slope = {se_true:.5f}")

# Coverage of b +/- 1.96 se over R simulated samples
hits, ses = np.zeros(5), np.zeros(5)
for r in range(R):
    x, y = simulate(n)
    b, se = all_se(x, y)
    hits += np.abs(b - b1) <= 1.96 * se
    ses += se
print(f"{'standard error':<20}{'mean se':>10}{'ratio to true':>15}"
      f"{'95% coverage':>14}")
for k in range(5):
    print(f"{names[k]:<20}{ses[k] / R:10.5f}{ses[k] / R / se_true:15.3f}"
          f"{hits[k] / R:14.3f}")
