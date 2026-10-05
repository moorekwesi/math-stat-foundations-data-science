# DEP 803 -- Chapter 8: Optimisation
# Pure Newton can diverge; damped Newton with backtracking is globally safe
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

f = lambda x: np.sqrt(1 + x**2)          # convex, minimum at x = 0
g = lambda x: x / np.sqrt(1 + x**2)
h = lambda x: (1 + x**2) ** -1.5


def newton(x, damped, iters=6):
    out = [x]
    for k in range(iters):
        d = -g(x) / h(x)                 # Newton direction (= -x - x^3 here)
        t = 1.0
        if damped:                       # Armijo backtracking
            while f(x + t * d) > f(x) + 1e-4 * t * g(x) * d:
                t *= 0.5
        x = x + t * d
        out.append(x)
    return out


for x0 in [0.5, 1.5]:
    for damped in [False, True]:
        xs = newton(x0, damped)
        lab = "damped" if damped else "pure  "
        print(f"x0 = {x0}, {lab}: " + " ".join(f"{v:9.2e}" for v in xs))
