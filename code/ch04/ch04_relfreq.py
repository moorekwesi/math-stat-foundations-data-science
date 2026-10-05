# DEP 803 -- Chapter 4: Probability Foundations, Random Variables and Transformations
# Relative frequencies of an event settle down as the number of trials grows.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
n_max, n_paths = 10_000, 5
rolls = rng.integers(1, 7, size=(n_paths, n_max))      # fair six-sided die
is_six = (rolls == 6)
n = np.arange(1, n_max + 1)
running = np.cumsum(is_six, axis=1) / n                 # relative frequency path

print("Relative frequency of the event {six} (true value 1/6 = 0.1667)")
print(f"{'n':>7}" + "".join(f"  path{j + 1}" for j in range(n_paths)))
for k in [10, 100, 1_000, 10_000]:
    row = "".join(f"{running[j, k - 1]:8.4f}" for j in range(n_paths))
    print(f"{k:>7}" + row)

# Event algebra on the same sample: A = {even}, B = {>= 4}
x = rolls[0]
A, B = (x % 2 == 0), (x >= 4)
pA, pB, pAB, pAuB = A.mean(), B.mean(), (A & B).mean(), (A | B).mean()
print("\nAddition rule on path 1 (relative frequencies obey it exactly):")
print(f"  f(A) + f(B) - f(A and B) = {pA + pB - pAB:.4f}")
print(f"  f(A or B)                = {pAuB:.4f}   (true 4/6 = {4 / 6:.4f})")

fig, ax = plt.subplots(figsize=(6, 3.8))
for j in range(n_paths):
    ax.plot(n, running[j], lw=1)
ax.axhline(1 / 6, color="k", ls="--", lw=1, label="P(six) = 1/6")
ax.set_xscale("log")
ax.set(xlabel="number of rolls n (log scale)", ylabel="relative frequency of a six")
ax.set_ylim(0, 0.45)
ax.legend(frameon=False)
ax.grid(alpha=0.3)
plt.savefig("ch04_relfreq.pdf", bbox_inches="tight")
