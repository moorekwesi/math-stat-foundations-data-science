# DEP 803 -- Chapter 0: Mathematical Refresher
# Lab 0, part 2: NumPy arrays, vectorised arithmetic and a first plot.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
# simulated monthly sales (GHS thousand) for 12 months -- not real data
sales = np.round(50 + 2 * np.arange(12) + rng.normal(0, 4, 12), 1)
print("sales:", sales)
print("shape:", sales.shape, " mean:", sales.mean().round(2),
      " max:", sales.max(), " in month", sales.argmax() + 1)

# vectorised arithmetic: one line acts on every element, no loop needed
vat = 0.15 * sales
growth = np.diff(sales) / sales[:-1]
print("VAT at 15% (first 3):", vat[:3].round(2))
print("months with falling sales:", np.where(growth < 0)[0] + 2)
print("average monthly growth: {:.2%}".format(growth.mean()))

# lists do not behave like vectors; arrays do
print("[1, 2] * 2 =", [1, 2] * 2, "   np.array([1, 2]) * 2 =", np.array([1, 2]) * 2)

# a 2-D array is a matrix
X = np.array([[1.0, 2.0], [3.0, 4.0]])
print("X @ X =\n", X @ X)

months = np.arange(1, 13)
slope, intercept = np.polyfit(months, sales, 1)
print(f"fitted trend: sales = {intercept:.2f} + {slope:.2f} * month")
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(months, sales, "o-", label="simulated sales")
ax.plot(months, intercept + slope * months, "--", label="straight-line trend")
ax.set_xlabel("month")
ax.set_ylabel("sales (GHS thousand)")
ax.legend(frameon=False, fontsize=8)
plt.savefig("ch00_lab0.pdf", bbox_inches="tight")
