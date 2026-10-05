# DEP 803 -- Chapter 4: Probability Foundations, Random Variables and Transformations
# Matching problem: inclusion-exclusion formula versus simulation.
# (c) 2026 Stephen E. Moore. MIT Licence.
import math
import numpy as np

rng = np.random.default_rng(803)


def p_match_exact(n):
    """P(at least one fixed point of a random permutation of n items)."""
    return sum((-1) ** (k + 1) / math.factorial(k) for k in range(1, n + 1))


def p_match_sim(n, reps):
    """Monte Carlo estimate using random permutations (vectorised)."""
    perms = np.argsort(rng.random((reps, n)), axis=1)
    return np.mean((perms == np.arange(n)).any(axis=1))


print("n enumerators are randomly assigned to n districts.")
print("Event: at least one enumerator is sent to his or her home district.\n")
print(f"{'n':>4} {'incl-excl':>11} {'simulated':>11} {'1 - 1/e':>9}")
for n in [2, 3, 5, 10, 50]:
    print(f"{n:>4} {p_match_exact(n):11.6f} {p_match_sim(n, 100_000):11.6f}"
          f" {1 - math.exp(-1):9.6f}")

# Continuity from below: A_n = {first six occurs within n rolls} increases
print("\nContinuity: P(first six within n rolls) = 1 - (5/6)^n increases to 1")
for n in [1, 5, 10, 25, 50]:
    print(f"  n = {n:>2}: {1 - (5 / 6) ** n:.6f}")
