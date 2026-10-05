# DEP 803 -- Chapter 6: Expectation, Covariance, Limit Theorems and Concentration
# Heavy tails slow the CLT: coverage of mean intervals for Pareto incomes.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)
reps = 2000
sizes = [50, 200, 1000, 5000]
print("Coverage of xbar +- 1.96 s/sqrt(n) for Pareto(alpha) incomes, x_m = 1")
print(f"{'alpha':>6}{'mean':>7}" + "".join(f"{'n=' + str(n):>9}" for n in sizes))
Zs = {}
for a in [1.5, 2.5, 4.0, 8.0]:
    mu = a / (a - 1)
    row = ""
    for n in sizes:
        x = rng.pareto(a, size=(reps, n)) + 1.0
        xbar, s = x.mean(axis=1), x.std(axis=1, ddof=1)
        z = (xbar - mu) / (s / np.sqrt(n))
        row += f"{np.mean(np.abs(z) <= 1.96):9.3f}"
        if n == 1000:
            Zs[a] = z
    print(f"{a:6.1f}{mu:7.3f}{row}")
print("(alpha = 1.5: infinite variance, the classical CLT does not apply)")

plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(figsize=(6, 3.8))
t = np.linspace(-6, 4, 400)
for a, col in [(1.5, "firebrick"), (2.5, "darkorange"), (8.0, "navy")]:
    ax.hist(Zs[a], bins=100, range=(-6, 4), density=True, histtype="step",
            lw=1.4, color=col, label=f"alpha = {a}")
ax.plot(t, stats.norm.pdf(t), "k--", lw=1.2, label="N(0,1)")
ax.set_xlabel("studentised mean, n = 1000")
ax.set_ylabel("density")
ax.legend(frameon=False)
plt.savefig("ch06_pareto.pdf", bbox_inches="tight")
