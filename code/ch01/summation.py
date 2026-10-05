# DEP 803 -- Chapter 1: Notation, Vectors, Matrices and Computational Precision
# Rounding error in long sums: naive loop, Kahan, pairwise (NumPy) and fsum.
# (c) 2026 Stephen E. Moore. MIT Licence.
import math
import numpy as np


def naive_sum(x, zero=0.0):
    s = zero
    for xi in x:
        s += xi
    return s


def kahan_sum(x, zero=0.0):
    """Compensated summation (Kahan, 1965)."""
    s, c = zero, zero
    for xi in x:
        y = xi - c            # correct the next term by the lost low part
        t = s + y             # big + small: low-order digits of y are lost
        c = (t - s) - y       # recover what was lost (algebraically zero)
        s = t
    return s


n = 1_000_000
x = np.full(n, 0.1)
exact = 100_000.0                       # the intended sum of the decimal 0.1s
print(f"summing 0.1 one million times (target {exact})")
for name, val in [("naive loop", naive_sum(x)), ("Kahan", kahan_sum(x)),
                  ("np.sum (pairwise)", np.sum(x)), ("math.fsum", math.fsum(x))]:
    print(f"  {name:18s} {val:.10f}   error {val - exact: .3e}")

x32 = x.astype(np.float32)
print("same data in float32:")
print(f"  {'naive loop':18s} {float(naive_sum(x32, np.float32(0))):.4f}")
print(f"  {'Kahan':18s} {float(kahan_sum(x32, np.float32(0))):.4f}")
print(f"  {'np.sum (pairwise)':18s} {float(np.sum(x32)):.4f}")

# order matters: a large total plus many small amounts (e.g. a tax ledger)
amounts = np.r_[1e16, np.ones(10_000)]
print("1e16 then 10,000 ones, left to right:", naive_sum(amounts) - 1e16)
print("small ones first                    :", naive_sum(amounts[::-1]) - 1e16)
print("math.fsum (correctly rounded)       :", math.fsum(amounts) - 1e16)
