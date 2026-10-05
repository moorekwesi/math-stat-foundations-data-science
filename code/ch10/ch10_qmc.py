# DEP 803 -- Chapter 10: Numerical Integration and Monte Carlo Simulation
# Quasi-Monte Carlo (scrambled Sobol and Halton) versus plain Monte Carlo.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import qmc

rng = np.random.default_rng(803)
d, reps = 5, 30
# smooth test integrand on [0,1]^d with a known integral
f = lambda u: np.prod(1 + (u - 0.5) * np.arange(1, d + 1) / d, axis=1) ** 2
exact = np.prod(1 + (np.arange(1, d + 1) / d) ** 2 / 12)

ms = np.arange(6, 15)
rmse = {"MC": [], "Sobol": [], "Halton": []}
for mm in ms:
    n = 2**mm
    err = {k: [] for k in rmse}
    for r in range(reps):
        err["MC"].append(f(rng.random((n, d))).mean() - exact)
        sob = qmc.Sobol(d, scramble=True, seed=rng).random_base2(mm)
        err["Sobol"].append(f(sob).mean() - exact)
        hal = qmc.Halton(d, scramble=True, seed=rng).random(n)
        err["Halton"].append(f(hal).mean() - exact)
    for k in rmse:
        rmse[k].append(np.sqrt(np.mean(np.square(err[k]))))

print(f"d = {d}, RMSE over {reps} randomisations")
print("      n        MC       Sobol      Halton")
for i, mm in enumerate(ms):
    print(f"{2**mm:7d}  {rmse['MC'][i]:.2e}  {rmse['Sobol'][i]:.2e}"
          f"  {rmse['Halton'][i]:.2e}")
for k in rmse:
    sl = np.polyfit(ms * np.log(2), np.log(rmse[k]), 1)[0]
    print(f"fitted rate n^({sl:.2f}) for {k}")
pts = qmc.Sobol(2, scramble=False).random_base2(8)
dis_s, dis_u = qmc.discrepancy(pts), qmc.discrepancy(rng.random((256, 2)))
print(f"centred L2 discrepancy, 256 points in 2-d: Sobol {dis_s:.2e},"
      f" pseudo-random {dis_u:.2e}")

fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.6))
axs[0].scatter(*rng.random((256, 2)).T, s=6, color="navy")
axs[0].set_title("256 pseudo-random points")
axs[1].scatter(*pts.T, s=6, color="darkorange")
axs[1].set_title("256 Sobol points")
for ax in axs:
    ax.set_aspect("equal")
    ax.set_xlabel("u1")
    ax.set_ylabel("u2")
plt.tight_layout()
plt.savefig("ch10_qmc_points.pdf", bbox_inches="tight")

fig, ax = plt.subplots(figsize=(6, 3.8))
for k, mk in zip(rmse, ["o", "s", "^"]):
    ax.loglog(2.0**ms, rmse[k], mk + "-", ms=4, label=k)
ax.loglog(2.0**ms, rmse["MC"][0] * (2.0 ** (ms - ms[0])) ** -0.5, "k:",
          label="slope -1/2")
ax.set_xlabel("number of points n")
ax.set_ylabel("RMSE")
ax.legend()
plt.savefig("ch10_qmc_rmse.pdf", bbox_inches="tight")
