# DEP 803 -- Chapter 10: Numerical Integration and Monte Carlo Simulation
# Gibbs sampling for a bivariate normal; autocorrelation and effective size.
# (c) 2026 Stephen E. Moore. MIT Licence.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # tiny matrices: one thread
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)


def gibbs(rho, n_iter, start=(-4.0, 4.0)):
    """Alternate x1 | x2 ~ N(rho x2, 1-rho^2) and x2 | x1 ~ N(rho x1, 1-rho^2)."""
    x = np.empty((n_iter, 2))
    x1, x2 = start
    sd = np.sqrt(1 - rho**2)
    z = rng.standard_normal((n_iter, 2))
    for t in range(n_iter):
        x1 = rho * x2 + sd * z[t, 0]
        x2 = rho * x1 + sd * z[t, 1]
        x[t] = x1, x2
    return x


def ess(x):                             # Geyer initial-positive-sequence ESS
    x = x - x.mean()
    m = len(x)
    f = np.fft.rfft(x, 2 * m)
    acf = np.fft.irfft(f * np.conj(f))[:m] / (x @ x)
    s = 0.0
    for k in range(0, m - 1, 2):
        if acf[k] + acf[k + 1] < 0:
            break
        s += acf[k] + acf[k + 1]
    return m / (2 * s - 1)


n_iter, burn = 20_000, 500
print(" rho   mean(x1)  var(x1)  corr   lag-1 acf  theory rho^2"
      "   ESS    theory ESS")
chains = {}
for rho in (0.5, 0.9, 0.99):
    x = gibbs(rho, n_iter)
    chains[rho] = x
    x1 = x[burn:, 0]
    a1 = np.corrcoef(x1[:-1], x1[1:])[0, 1]
    r2 = rho**2
    th = len(x1) * (1 - r2) / (1 + r2)        # AR(1) with coefficient rho^2
    print(f"{rho:5.2f}  {x1.mean():8.3f}  {x1.var():7.3f}  "
          f"{np.corrcoef(x[burn:].T)[0, 1]:5.3f}  {a1:9.3f}  {r2:12.3f}"
          f"  {ess(x1):6.0f}  {th:9.0f}")

fig, axs = plt.subplots(1, 2, figsize=(7.5, 3.6))
x = chains[0.9]
axs[0].plot(*x[:40].T, "-o", ms=2.5, lw=0.7, color="navy")
axs[0].scatter(*x[burn:burn + 2000].T, s=2, color="gray", alpha=0.4)
axs[0].set_title("first 40 Gibbs steps, rho = 0.9", fontsize=10)
axs[0].set_xlabel("x1")
axs[0].set_ylabel("x2")
for rho, col in zip((0.5, 0.99), ("darkorange", "firebrick")):
    axs[1].plot(chains[rho][:600, 0], lw=0.7, color=col, label=f"rho = {rho}")
axs[1].set_xlabel("iteration")
axs[1].set_ylabel("x1")
axs[1].set_title("trace plots", fontsize=10)
axs[1].legend(fontsize=8)
plt.tight_layout()
plt.savefig("ch10_gibbs.pdf", bbox_inches="tight")
