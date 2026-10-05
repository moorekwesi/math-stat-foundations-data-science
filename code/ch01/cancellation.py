# DEP 803 -- Chapter 1: Notation, Vectors, Matrices and Computational Precision
# Catastrophic cancellation and how algebraic rewriting cures it.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

# (exp(x) - 1)/x -> 1 as x -> 0: naive formula versus expm1
print(f"{'x':>8s} {'naive (e^x-1)/x':>18s} {'expm1(x)/x':>18s}")
for x in [1e-4, 1e-8, 1e-12, 1e-15, 1e-16]:
    print(f"{x:8.0e} {(np.exp(x) - 1) / x:18.15f} {np.expm1(x) / x:18.15f}")

# (1 - cos x)/x^2 -> 1/2: rewrite as 2 sin^2(x/2)/x^2
x = 1.2e-8
print("\n(1-cos x)/x^2      at x=1.2e-8:", (1 - np.cos(x)) / x**2)
print("2 sin^2(x/2)/x^2   at x=1.2e-8:", 2 * np.sin(x / 2)**2 / x**2)

# roots of x^2 + b x + c with b >> c: textbook versus stable formula
b, c = 1e8, 1.0
disc = np.sqrt(b * b - 4 * c)
r_naive = (-b + disc) / 2
q = -0.5 * (b + np.sign(b) * disc)
r_stable = c / q
print("\nsmall root, textbook formula :", r_naive)
print("small root, stable c/q       :", r_stable, "  (true value about -1e-8)")
