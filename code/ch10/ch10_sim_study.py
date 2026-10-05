# DEP 803 -- Chapter 10: Numerical Integration and Monte Carlo Simulation
# ADEMP simulation study: OLS confidence intervals under heteroskedasticity.
# (c) 2026 Stephen E. Moore. MIT Licence.
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")  # tiny matrices: one thread
import numpy as np

# Aims: do classical and HC1 (White) 95% CIs for the slope achieve coverage?
# Data-generating mechanisms: y = 1 + 0.5 x + e, x ~ Exp(1), sd(e) = x^gamma
# Estimand: slope beta1 = 0.5.  Methods: OLS + classical / HC1 s.e.
# Performance: bias, empirical s.e., coverage, each with its MC s.e.
rng = np.random.default_rng(803)
n_sim, beta1 = 4000, 0.5
print(f"n_sim = {n_sim} per scenario; Monte Carlo s.e. in parentheses")
print(" n  gamma  bias            emp.se   cover(classical)  cover(HC1)")
for n in (50, 400):
    for gam in (0.0, 1.0):
        x = rng.exponential(1.0, (n_sim, n))
        e = rng.standard_normal((n_sim, n)) * x**gam
        y = 1 + beta1 * x + e
        xc = x - x.mean(axis=1, keepdims=True)
        sxx = (xc**2).sum(axis=1)
        b1 = (xc * y).sum(axis=1) / sxx
        b0 = y.mean(axis=1) - b1 * x.mean(axis=1)
        res = y - b0[:, None] - b1[:, None] * x
        se_cl = np.sqrt((res**2).sum(axis=1) / (n - 2) / sxx)
        se_hc = np.sqrt(n / (n - 2) * (xc**2 * res**2).sum(axis=1)) / sxx
        bias = b1.mean() - beta1
        emp = b1.std(ddof=1)
        out = f"{n:3d}  {gam:4.1f}  {bias: .4f} ({emp / np.sqrt(n_sim):.4f})"
        out += f"  {emp:.4f}"
        for se in (se_cl, se_hc):
            cov = np.mean(np.abs(b1 - beta1) <= 1.96 * se)
            mcse = np.sqrt(cov * (1 - cov) / n_sim)
            out += f"   {cov:.3f} ({mcse:.4f})"
        print(out)
# How many replications for a coverage MC s.e. of 0.005 at 95%?
print(f"n_sim needed for MC s.e. 0.005 at coverage 0.95: "
      f"{int(np.ceil(0.95 * 0.05 / 0.005**2))}")
