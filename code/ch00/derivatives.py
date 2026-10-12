# DEP 803 -- Chapter 0: Mathematical Refresher
# Derivatives as rates of change: marginal cost, and profit maximisation.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt


def cost(q):
    return 100 + 4 * q + 0.5 * q**2      # total cost, GHS


def revenue(q):
    return 40 * q - q**2                  # total revenue, GHS


def profit(q):
    return revenue(q) - cost(q)


# difference quotients approach the derivative C'(10) = 4 + 10 = 14
q0 = 10.0
for h in [1.0, 0.1, 0.01, 0.001]:
    slope = (cost(q0 + h) - cost(q0)) / h
    print(f"h = {h:6.3f}   [C(10+h) - C(10)] / h = {slope:.4f}")
print("exact marginal cost C'(10) = 4 + 1.0*10 =", 4 + q0)

# profit maximisation: FOC 36 - 3q = 0  gives  q* = 12
q = np.linspace(0, 24, 241)
best = q[np.argmax(profit(q))]
print(f"grid search: best q = {best:.1f}, profit = {profit(best):.1f}")
print("by calculus: q* = 36/3 = 12, profit =", profit(12.0))
print("marginal revenue = marginal cost at q*:", 40 - 2 * 12, "=", 4 + 12)

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(q, revenue(q), label="revenue R(q)")
ax.plot(q, cost(q), label="cost C(q)")
ax.plot(q, profit(q), label="profit R(q) - C(q)", lw=2)
ax.plot(q, profit(12) + 0 * q, "k:", lw=0.8)
ax.plot([12, 12], [-100, profit(12)], "k:", lw=0.8)
ax.annotate("q* = 12: slope of profit is zero", xy=(12, profit(12)),
            xytext=(13.5, 200), arrowprops=dict(arrowstyle="->"), fontsize=8)
ax.set_xlabel("quantity q")
ax.set_ylabel("GHS")
ax.axhline(0, color="gray", lw=0.5)
ax.legend(frameon=False, fontsize=8, loc="upper left")
plt.savefig("ch00_profit.pdf", bbox_inches="tight")
