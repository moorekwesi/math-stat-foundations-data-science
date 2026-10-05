# DEP 803 -- Chapter 9: Constrained Optimisation and Lagrange Multipliers
# Least-cost nutritious food basket (Stigler diet LP) with simulated prices.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import linprog

# Stylised nutrient content per 100 g (teaching values, NOT a food table)
foods = ["maize flour", "rice", "gari", "cowpea", "groundnut", "dried fish",
         "eggs", "leafy greens", "red palm oil", "milk powder", "plantain"]
nutr = ["energy kcal", "protein g", "fat g", "iron mg", "calcium mg",
        "vit A ug"]
N = np.array([
    # kcal  prot   fat  iron   calc   vitA
    [360,   9.0,  4.0,  3.5,     7,     10],   # maize flour
    [360,   7.0,  0.6,  0.8,    10,      0],   # rice
    [360,   1.2,  0.5,  1.5,    40,      0],   # gari
    [340,  23.0,  1.3,  8.0,   110,      2],   # cowpea
    [570,  25.0, 49.0,  4.6,    90,      0],   # groundnut
    [330,  60.0,  6.0,  5.0,  1500,     50],   # dried fish
    [145,  12.5, 10.0,  1.8,    55,    160],   # eggs
    [ 40,   3.5,  0.7,  2.5,   200,    300],   # leafy greens
    [880,   0.0, 99.0,  0.0,     0,   5000],   # red palm oil
    [500,  26.0, 27.0,  0.5,   900,    300],   # milk powder
    [120,   1.3,  0.4,  0.6,     3,     50]]).T  # plantain
price = np.array([0.9, 1.6, 0.8, 2.0, 2.6, 4.5, 3.5, 0.6, 2.2, 6.0, 0.7])
req = np.array([2300, 50, 45, 18, 1000, 600])     # daily requirement
upper = np.array([5, 4, 5, 3, 1.5, 1, 2, 3, 0.6, 1, 4])  # max 100 g units

# min price'x  s.t.  N x >= req  (written as -N x <= -req),  0 <= x <= upper
res = linprog(price, A_ub=-N, b_ub=-req, bounds=list(zip(0 * upper, upper)),
              method="highs")
x = res.x
print("Least-cost basket (simulated prices), solver status =", res.status)
for f, xi, pi in zip(foods, x, price):
    if xi > 1e-9:
        print("  %-13s %5.0f g   cost GHS %5.2f" % (f, 100 * xi, pi * xi))
print("  daily cost = GHS %.3f" % res.fun)

y = -res.ineqlin.marginals                  # shadow prices (>= 0)
print("\n  nutrient        required  supplied  shadow price (GHS/unit)")
for n, r, s, yi in zip(nutr, req, N @ x, y):
    print("  %-13s %9.0f %9.1f   %.5f" % (n, r, s, yi))
z = -res.upper.marginals                    # value of relaxing upper bounds
print("  binding upper bounds:", [f for f, zi in zip(foods, z) if zi > 1e-9])

# dual LP: max req'y - upper'z  s.t.  N'y - z <= price,  y, z >= 0
nf, nn = len(foods), len(nutr)
dual = linprog(np.r_[-req, upper], A_ub=np.hstack([N.T, -np.eye(nf)]),
               b_ub=price, bounds=[(0, None)] * (nn + nf), method="highs")
print("\nDual LP optimum = %.3f  (primal %.3f): strong duality" %
      (-dual.fun, res.fun))
rc = price - N.T @ y + z
print("reduced costs (GHS per 100 g) of foods not in the basket:")
print("  " + ", ".join("%s %.2f" % (f, r) for f, r in zip(foods, rc)
                       if r > 1e-6))

# parametric analysis: cost as a function of the protein requirement
prot = np.linspace(20, 140, 121)
cost = []
for pr in prot:
    r2 = req.copy()
    r2[1] = pr
    cost.append(linprog(price, A_ub=-N, b_ub=-r2,
                        bounds=list(zip(0 * upper, upper)),
                        method="highs").fun)
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(prot, cost, lw=2)
ax.axvline(req[1], ls="--", color="grey")
ax.plot(req[1], res.fun, "o", color="C3")
ax.annotate("slope = shadow price of protein\n= GHS %.4f per g" % y[1],
            (req[1] + 4, res.fun + 2.0), fontsize=8)
ax.set_xlabel("protein requirement (g per day)")
ax.set_ylabel("least cost (GHS per day)")
ax.set_title("Parametric LP: cost of a nutritious diet", fontsize=10)
ax.grid(alpha=0.3)
plt.savefig("ch09_diet_lp.pdf", bbox_inches="tight")
