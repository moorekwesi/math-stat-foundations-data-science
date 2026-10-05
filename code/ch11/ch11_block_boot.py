# DEP 803 -- Chapter 11: Resampling: Bootstrap, Permutation Tests, Intervals
# Moving block bootstrap for the mean of an autocorrelated AR(1) series.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
T, phi, B = 240, 0.6, 2000      # e.g. 20 years of monthly growth rates


def ar1(size):
    e = rng.normal(0, 1, size=size + (T + 100,))
    x = np.zeros_like(e)
    for t in range(1, e.shape[-1]):
        x[..., t] = phi * x[..., t - 1] + e[..., t]
    return x[..., 100:]                                  # drop burn-in


def mbb_means(x, ell):
    """B moving-block-bootstrap means with block length ell."""
    nblk = int(np.ceil(T / ell))
    starts = rng.integers(0, T - ell + 1, size=(B, nblk))
    idx = (starts[:, :, None] + np.arange(ell)).reshape(B, -1)[:, :T]
    return x[idx].mean(axis=1)


x = ar1(())
se_true = ar1((5000,)).mean(axis=1).std(ddof=1)
se_lr = 1 / (1 - phi) / np.sqrt(T)                       # long-run formula
se_iid = x[rng.integers(0, T, size=(B, T))].mean(axis=1).std(ddof=1)
print(f"T = {T}, phi = {phi}: sample mean = {x.mean():.3f}")
print(f"true sd of mean (MC) = {se_true:.4f}; asymptotic = {se_lr:.4f}")
print(f"iid bootstrap se = {se_iid:.4f}  (ignores dependence)")
ells = np.array([1, 2, 4, 6, 8, 12, 16, 24, 32, 48])
se_mbb = np.array([mbb_means(x, ell).std(ddof=1) for ell in ells])
print(f"{'block length':>12}{'MBB se':>9}")
for ell, s in zip(ells, se_mbb):
    print(f"{ell:12d}{s:9.4f}")
print(f"rule of thumb ell ~ T^(1/3) = {T**(1 / 3):.1f}")

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(ells, se_mbb, "o-", color="steelblue", label="moving block bootstrap")
ax.axhline(se_true, color="black", ls="--", lw=1, label="true sd (Monte Carlo)")
ax.axhline(se_iid, color="firebrick", ls=":", lw=1.2, label="iid bootstrap")
ax.set_xscale("log", base=2)
ax.set_xlabel(r"block length $\ell$")
ax.set_ylabel("standard error of the mean")
ax.legend(frameon=False, loc="lower right")
plt.savefig("ch11_block.pdf", bbox_inches="tight")
