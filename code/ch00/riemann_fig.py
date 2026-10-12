# DEP 803 -- Chapter 0: Mathematical Refresher
# Picture of a midpoint Riemann sum for f(x) = 3x^2 + 1 on [0, 2].
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt


def f(x):
    return 3 * x**2 + 1


def riemann(func, a, b, n):
    dx = (b - a) / n
    return np.sum(func(a + dx * (np.arange(n) + 0.5)) * dx)


# picture of a Riemann sum with n = 6 strips
xs = np.linspace(0, 2, 200)
n = 6
dx = 2 / n
mids = dx * (np.arange(n) + 0.5)
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.bar(mids, f(mids), width=dx, alpha=0.35, edgecolor="k", label="6 midpoint strips")
ax.plot(xs, f(xs), lw=2, label="f(x) = 3x^2 + 1")
ax.set_xlabel("x")
ax.set_ylabel("f(x)")
total = riemann(f, 0, 2, n)
ax.set_title(f"strip total = {total:.4f};  exact area = 10", fontsize=9)
ax.legend(frameon=False, fontsize=8, loc="upper left")
plt.savefig("ch00_riemann.pdf", bbox_inches="tight")
print("figure ch00_riemann.pdf written")
