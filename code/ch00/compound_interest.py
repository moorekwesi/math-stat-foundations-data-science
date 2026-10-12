# DEP 803 -- Chapter 0: Mathematical Refresher
# Compound interest for different compounding frequencies, and the limit e.
# (c) 2026 Stephen E. Moore. MIT Licence.
import math

principal = 1000.0      # GHS
rate = 0.20             # 20% nominal annual rate
years = 2

print("Value of GHS 1000 after 2 years at a nominal 20% a year")
print(f"{'compounding':>14s} {'m':>6s} {'value (GHS)':>12s}")
for name, m in [("annual", 1), ("semi-annual", 2), ("quarterly", 4),
                ("monthly", 12), ("daily", 365)]:
    value = principal * (1 + rate / m) ** (m * years)
    print(f"{name:>14s} {m:6d} {value:12.2f}")
cont = principal * math.exp(rate * years)
print(f"{'continuous':>14s} {'inf':>6s} {cont:12.2f}")

print("\n(1 + 1/m)^m approaches e = 2.718281828...")
for m in [1, 10, 100, 10_000, 1_000_000]:
    print(f"m = {m:>9,d}   (1 + 1/m)^m = {(1 + 1 / m) ** m:.8f}")

# solving for time with logarithms: how long until money doubles?
for r in [0.05, 0.10, 0.20]:
    t_exact = math.log(2) / math.log(1 + r)
    print(f"r = {r:.2f}: doubling time = {t_exact:5.2f} years,"
          f" rule of 70 gives {70 / (100 * r):5.2f}")
