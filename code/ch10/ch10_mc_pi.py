# DEP 803 -- Chapter 10: Numerical Integration and Monte Carlo Simulation
# Monte Carlo estimate of pi with standard errors, CIs and a coverage check.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)


def mc_pi(n, rng):
    u = rng.random((n, 2))
    h = 4.0 * (u[:, 0] ** 2 + u[:, 1] ** 2 <= 1.0)  # h(U) = 4 * indicator
    return h.mean(), h.std(ddof=1) / np.sqrt(n)


print("        n    estimate     MC s.e.   95% CI                |error|")
for n in 10 ** np.arange(2, 8):
    est, se = mc_pi(n, rng)
    print(f"{n:9d}  {est:10.6f}  {se:10.6f}   [{est - 1.96 * se:8.5f},"
          f" {est + 1.96 * se:8.5f}]  {abs(est - np.pi):.2e}")

# Do nominal 95% intervals cover pi 95% of the time? (R = 2000 replications)
R, n = 2000, 1000
u = rng.random((R, n, 2))
h = 4.0 * (u[..., 0] ** 2 + u[..., 1] ** 2 <= 1.0)
est, se = h.mean(axis=1), h.std(axis=1, ddof=1) / np.sqrt(n)
cover = np.mean(np.abs(est - np.pi) <= 1.96 * se)
print(f"\ncoverage of 95% CI, n={n}: {cover:.3f}"
      f"  (MC s.e. of coverage {np.sqrt(cover * (1 - cover) / R):.3f})")
print(f"theoretical sd of h(U): {4 * np.sqrt(np.pi / 4 * (1 - np.pi / 4)):.4f};"
      f" mean estimated sd: {np.mean(se) * np.sqrt(n):.4f}")

# Figure: running estimate with a 95% band
n = 20000
u = rng.random((n, 2))
h = 4.0 * (u[:, 0] ** 2 + u[:, 1] ** 2 <= 1.0)
k = np.arange(1, n + 1)
run = np.cumsum(h) / k
sd = np.sqrt(np.pi * (4 - np.pi))
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(k, run, color="navy", lw=1, label="running MC estimate")
ax.fill_between(k, np.pi - 1.96 * sd / np.sqrt(k), np.pi + 1.96 * sd / np.sqrt(k),
                color="darkorange", alpha=0.25, label="pi +/- 1.96 sd/sqrt(n)")
ax.axhline(np.pi, color="black", lw=0.8)
ax.set_xscale("log")
ax.set_ylim(2.6, 3.7)
ax.set_xlabel("number of draws n")
ax.set_ylabel("estimate of pi")
ax.legend()
plt.savefig("ch10_mc_pi.pdf", bbox_inches="tight")
