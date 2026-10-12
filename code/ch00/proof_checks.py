# DEP 803 -- Chapter 0: Mathematical Refresher
# Computers check examples and find counterexamples; only proofs cover all cases.
# (c) 2026 Stephen E. Moore. MIT Licence.


def is_prime(m):
    """True if m has no divisor d with 2 <= d <= sqrt(m)."""
    return m >= 2 and all(m % d for d in range(2, int(m**0.5) + 1))


# claim proved by induction: 1 + 3 + ... + (2n - 1) = n^2
ok = all(sum(2 * i - 1 for i in range(1, n + 1)) == n**2 for n in range(1, 1001))
print("sum of first n odd numbers equals n^2 for n = 1..1000:", ok)

# a tempting conjecture: n^2 + n + 41 is always prime
first_fail = next(n for n in range(0, 100) if not is_prime(n**2 + n + 41))
print("n^2 + n + 41 is prime for n = 0 ..", first_fail - 1)
m = first_fail**2 + first_fail + 41
print(f"counterexample: n = {first_fail} gives {m} = 41 * {m // 41}")

# 'for every x, x^2 > x' -- search a grid for counterexamples
grid = [k / 4 for k in range(-8, 9)]
bad = [x for x in grid if not x**2 > x]
print("grid points where x^2 > x fails:", bad)

# the accounting equation A = L + E survives every double-entry transaction
assets, liabilities, equity = 0, 0, 0
transactions = [("owner invests cash", 5000, 0, 5000),
                ("bank loan received", 2000, 2000, 0),
                ("buy equipment for cash", 0, 0, 0),
                ("sell goods for cash (profit 300)", 300, 0, 300),
                ("repay part of loan", -500, -500, 0)]
for name, dA, dL, dE in transactions:
    assets, liabilities, equity = assets + dA, liabilities + dL, equity + dE
    print(f"{name:34s} A = {assets:5d}  L + E = {liabilities + equity:5d}")
