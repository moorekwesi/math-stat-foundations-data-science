# DEP 803 -- Chapter 0: Mathematical Refresher
# Simple versus log returns for a (made-up) share price; ln(1 + x) ~ x.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

price = np.array([50.0, 55.0, 50.0, 52.0, 49.4])   # illustrative prices, GHS
simple = price[1:] / price[:-1] - 1                  # (P_t - P_{t-1}) / P_{t-1}
logret = np.log(price[1:] / price[:-1])              # ln(P_t / P_{t-1})

print(" t  price   simple return   log return")
for t in range(1, price.size):
    print(f"{t:2d} {price[t]:6.2f} {simple[t - 1]:13.4f} {logret[t - 1]:12.4f}")
print("sum of simple returns        :", round(simple.sum(), 4))
print("sum of log returns           :", round(logret.sum(), 4))
print("ln(P_end / P_start)          :", round(np.log(price[-1] / price[0]), 4))
print("compound simple: prod(1+r)-1 :", round(np.prod(1 + simple) - 1, 4))

# the approximation ln(1 + x) ~ x is good for small x
for x in [0.01, 0.05, 0.10, 0.50]:
    lx = np.log1p(x)                                 # ln(1 + x)
    print(f"x = {x:4.2f}  ln(1+x) = {lx:.4f}  error = {x - lx:.4f}")
