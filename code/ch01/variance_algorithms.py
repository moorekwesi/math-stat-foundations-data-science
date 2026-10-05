# DEP 803 -- Chapter 1: Notation, Vectors, Matrices and Computational Precision
# One-pass (textbook), two-pass and Welford variance under a large offset.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)
n = 10_000
z = rng.standard_normal(n)
z = (z - z.mean()) / z.std(ddof=1)        # exact sample variance 1 (to ~1e-16)


def one_pass(x):
    """Textbook 'computing formula': (sum x^2 - n xbar^2)/(n - 1)."""
    s, s2 = x.sum(), (x * x).sum()
    return (s2 - s * s / x.size) / (x.size - 1)


def two_pass(x):
    d = x - x.mean()
    return (d @ d) / (x.size - 1)


def welford(x):
    """Single pass, numerically stable update (Welford, 1962)."""
    mean, m2 = 0.0, 0.0
    for k, xk in enumerate(x, start=1):
        delta = xk - mean
        mean += delta / k
        m2 += delta * (xk - mean)
    return m2 / (len(x) - 1)


offsets = [0, 1e2, 1e4, 1e6, 1e7, 1e8, 1e9]
print(f"{'offset':>8s} {'one-pass':>14s} {'two-pass':>12s} {'Welford':>12s}")
for c in offsets:
    x = c + z                             # e.g. incomes measured in cedis
    v = [one_pass(x), two_pass(x), welford(x.tolist())]
    print(f"{c:8.0e} {v[0]:14.6f} {v[1]:12.9f} {v[2]:12.9f}")
x32 = (1e4 + z).astype(np.float32)
print("float32, offset 1e4: one-pass = %.4f, two-pass = %.6f"
      % (one_pass(x32), two_pass(x32)))

