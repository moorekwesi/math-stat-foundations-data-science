# DEP 803 -- Chapter 10: Numerical Integration and Monte Carlo Simulation
# Tensor-product Gauss rules versus Monte Carlo as the dimension grows.
# (c) 2026 Stephen E. Moore. MIT Licence.
import itertools
import numpy as np

rng = np.random.default_rng(803)
g = lambda u: 0.5 * np.pi * np.sin(np.pi * u)   # integrates to 1 on [0,1]
m = 4                                           # Gauss nodes per dimension
t, w = np.polynomial.legendre.leggauss(m)
t, w = 0.5 * (t + 1), 0.5 * w                   # map nodes to [0,1]

print(" d   tensor nodes   tensor error   MC error (same n)")
for d in range(1, 9):
    nodes = np.array(list(itertools.product(t, repeat=d)))
    wts = np.prod(np.array(list(itertools.product(w, repeat=d))), axis=1)
    est = wts @ np.prod(g(nodes), axis=1)
    n = m**d
    mc = np.prod(g(rng.random((n, d))), axis=1).mean()
    print(f"{d:2d} {n:13d}   {abs(est - 1):12.2e}   {abs(mc - 1):12.2e}")

print("\n10-point tensor rule: nodes and time at 1e9 evaluations/second")
for d in (1, 2, 5, 10, 20, 50):
    secs = 10.0**d / 1e9
    yrs = secs / (3600 * 24 * 365.25)
    print(f" d = {d:2d}: n = 1e{d:<3d} time = {secs:8.1e} s = {yrs:8.1e} years")
