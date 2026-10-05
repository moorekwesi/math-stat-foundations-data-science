# DEP 803 -- Chapter 5: Common Distributions in Data Science and Econometrics
# Build chi-square, t and F variables from normals and compare with scipy.stats
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)
R, k, m = 200_000, 5, 10
Z = rng.standard_normal((R, k))
chi2_k = (Z**2).sum(axis=1)                      # chi-square(k)
chi2_m = (rng.standard_normal((R, m)) ** 2).sum(axis=1)
t_k = rng.standard_normal(R) / np.sqrt(chi2_k / k)   # Student t(k)
F_km = (chi2_k / k) / (chi2_m / m)                # F(k, m)

print("95th percentiles: simulated vs exact")
for name, x, d in [("chi2(5)", chi2_k, stats.chi2(k)),
                   ("t(5)", t_k, stats.t(k)),
                   ("F(5,10)", F_km, stats.f(k, m))]:
    print(f"  {name:8s} {np.quantile(x, 0.95):7.3f}  {d.ppf(0.95):7.3f}"
          f"   mean {x.mean():6.3f} vs {d.mean():6.3f}")

# t(k)^2 is F(1,k)
print(f"P(t(5)^2 > 3) = {(t_k**2 > 3).mean():.4f},  "
      f"F(1,5) tail = {stats.f(1, k).sf(3):.4f}")

# Cochran: (n-1) S^2 / sigma^2 ~ chi2(n-1) for normal samples of size n
n, sigma = 8, 2.0
X = rng.normal(5.0, sigma, size=(R, n))
q = (n - 1) * X.var(axis=1, ddof=1) / sigma**2
print(f"(n-1)S^2/sigma^2: mean {q.mean():.3f} (theory {n - 1}), "
      f"var {q.var():.3f} (theory {2 * (n - 1)})")
print(f"corr(Xbar, S^2) = {np.corrcoef(X.mean(1), X.var(1, ddof=1))[0, 1]:.4f}")

print("two-sided 5% critical values of t(df) versus z = 1.960")
print("  " + "  ".join(f"df={d}: {stats.t(d).ppf(0.975):.3f}"
                       for d in [2, 5, 10, 30, 120]))

fig, ax = plt.subplots(figsize=(6, 3.8))
x = np.linspace(-5, 5, 400)
ax.plot(x, stats.norm.pdf(x), "k", lw=2.5, label="N(0,1)")
for df, col in [(1, "#b22222"), (3, "#cc9900"), (10, "#1f4e79")]:
    lab = f"t({df})" + (" = Cauchy" if df == 1 else "")
    ax.plot(x, stats.t(df).pdf(x), lw=1.8, color=col, label=lab)
ax.set_xlabel("x")
ax.set_ylabel("density")
ax.legend(frameon=False)
plt.savefig("ch05_t_vs_normal.pdf", bbox_inches="tight")
