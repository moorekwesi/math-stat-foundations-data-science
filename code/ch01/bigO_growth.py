# DEP 803 -- Chapter 1: Notation, Vectors, Matrices and Computational Precision
# Growth rates behind O-notation and back-of-the-envelope run-time estimates.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
n = np.logspace(1, 6, 200)
curves = {"log n": np.log2(n), "n": n, "n log n": n * np.log2(n),
          "n^2": n**2, "n^3": n**3}
fig, ax = plt.subplots(figsize=(6, 3.8))
for lab, val in curves.items():
    ax.loglog(n, val, lw=2, label=lab)
ax.set_xlabel("problem size n")
ax.set_ylabel("operations (log scale)")
ax.set_title("Growth of common cost functions")
ax.legend(frameon=False, ncol=2)
plt.savefig("ch01_growth.pdf", bbox_inches="tight")

# order-of-magnitude reasoning: time at 1e9 operations per second
speed = 1e9
print("Estimated run time at 1e9 operations per second")
print(f"{'n':>8} {'n log n':>12} {'n^2':>12} {'n^3':>12}")
for m in [1e3, 1e5, 1e7]:
    t = [m * np.log2(m) / speed, m**2 / speed, m**3 / speed]
    print(f"{m:8.0e} {t[0]:11.1e}s {t[1]:11.1e}s {t[2]:11.1e}s")
print("one year is about %.2e seconds" % (365.25 * 24 * 3600))
