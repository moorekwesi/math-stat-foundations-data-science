# DEP 803 -- Chapter 6: Expectation, Covariance, Limit Theorems and Concentration
# Tail of a Bernoulli mean: exact, CLT, Chebyshev, Hoeffding and Chernoff (KL).
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def kl(a, p):
    """Kullback-Leibler divergence between Bernoulli(a) and Bernoulli(p)."""
    return a * np.log(a / p) + (1 - a) * np.log((1 - a) / (1 - p))


eps = 0.05
ns = np.arange(20, 2001, 10)
print(f"Upper tail P(Xbar - p >= {eps}) for Bernoulli(p) means")
for p in [0.5, 0.1]:
    print(f"\np = {p}")
    print(f"{'n':>6}{'exact':>11}{'CLT':>11}{'Chebyshev':>11}{'Hoeffding':>11}"
          f"{'Chernoff':>11}")
    for n in [100, 400, 1000, 2000]:
        k = int(np.ceil(n * (p + eps) - 1e-9))
        exact = stats.binom.sf(k - 1, n, p)
        clt = stats.norm.sf(eps * np.sqrt(n / (p * (1 - p))))
        cheb = min(1, p * (1 - p) / (n * eps**2))     # two-sided bound
        hoef = np.exp(-2 * n * eps**2)
        cher = np.exp(-n * kl(p + eps, p))
        print(f"{n:6d}{exact:11.2e}{clt:11.2e}{cheb:11.2e}{hoef:11.2e}"
              f"{cher:11.2e}")

p = 0.1
k = np.ceil(ns * (p + eps) - 1e-9)
plt.rcParams.update({"font.size": 10})
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.semilogy(ns, stats.binom.sf(k - 1, ns, p), "k", lw=2, label="exact")
ax.semilogy(ns, stats.norm.sf(eps * np.sqrt(ns / (p * (1 - p)))), "--",
            label="CLT approximation")
ax.semilogy(ns, np.minimum(1, p * (1 - p) / (ns * eps**2)), ":",
            label="Chebyshev")
ax.semilogy(ns, np.exp(-2 * ns * eps**2), "-.", label="Hoeffding")
ax.semilogy(ns, np.exp(-ns * kl(p + eps, p)), lw=1.5, label="Chernoff (KL)")
ax.set_xlabel("sample size n")
ax.set_ylabel("P(Xbar - p >= 0.05)")
ax.set_title("Bernoulli(0.1) mean: bounds versus exact tail")
ax.set_ylim(1e-12, 2)
ax.legend(frameon=False, fontsize=9)
plt.savefig("ch06_concentration.pdf", bbox_inches="tight")
