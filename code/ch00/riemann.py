# DEP 803 -- Chapter 0: Mathematical Refresher
# Riemann sums, consumer surplus and the present value of an income stream.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np


def riemann(f, a, b, n):
    """Midpoint Riemann sum of f over [a, b] with n equal strips."""
    dx = (b - a) / n
    mid = a + dx * (np.arange(n) + 0.5)
    return np.sum(f(mid) * dx)


f = lambda x: 3 * x**2 + 1               # integral over [0, 2] is 10
for n in [2, 4, 10, 100, 1000]:
    s = riemann(f, 0, 2, n)
    print(f"n = {n:5d} strips: sum = {s:.6f}   error = {10 - s:.2e}")

# consumer surplus for demand p = 20 - 2q at market price 10 (q = 5)
cs = riemann(lambda q: (20 - 2 * q) - 10, 0, 5, 1000)
print(f"\nconsumer surplus (numerical) = {cs:.4f}; triangle 0.5*5*10 = 25")

# present value of GHS 1000 a year paid continuously for 10 years at r = 5%
r, T, c = 0.05, 10, 1000
pv_num = riemann(lambda t: c * np.exp(-r * t), 0, T, 10_000)
pv_exact = c * (1 - np.exp(-r * T)) / r
print(f"PV of income stream: numerical {pv_num:.2f}, formula {pv_exact:.2f}")

# probability as area: density f(x) = 2x on [0, 1]
prob = riemann(lambda x: 2 * x, 0, 0.5, 1000)
print("P(X <= 0.5) for density 2x:", round(prob, 4))
