# DEP 803 -- Chapter 10: Numerical Integration and Monte Carlo Simulation
# Microsimulation of a targeted cash transfer on poverty, with uncertainty.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
# A simulated household survey (NOT real data): 8,000 households
H = 8000
size = rng.integers(1, 9, H)
rural = rng.random(H) < 0.55
logc = 7.55 - 0.35 * rural - 0.06 * size + rng.normal(0, 0.55, H)
cons = np.exp(logc)                               # per-capita consumption
zline = 900.0                                     # poverty line (per capita)


def fgt(c, alpha, w):
    g = np.maximum(1 - c / zline, 0.0)
    return np.sum(w * g**alpha * (c < zline)) / w.sum()


w = size.astype(float)                            # person weights
P0, P1 = fgt(cons, 0, w), fgt(cons, 1, w)
budget = 0.02 * np.sum(w * cons)                  # 2% of total consumption


def simulate(pmt_noise, mpc, take_up, rng):
    """One draw of the policy outcome given uncertain parameters."""
    score = logc + rng.normal(0, pmt_noise, H)    # proxy-means test score
    elig = score < np.quantile(score, 0.25)       # bottom 25% by PMT score
    enrol = elig & (rng.random(H) < take_up)
    per_cap = budget / np.sum(w * enrol)          # budget split per person
    new = cons + mpc * per_cap * enrol
    return fgt(new, 0, w), fgt(new, 1, w)


# Outer loop: parameter uncertainty; inner noise: targeting and take-up
S = 1000
draws = np.empty((S, 2))
for s in range(S):
    pmt_noise = rng.uniform(0.2, 0.6)             # targeting error
    mpc = rng.uniform(0.7, 1.0)                   # share consumed
    take_up = rng.beta(16, 4)                     # mean 0.8
    draws[s] = simulate(pmt_noise, mpc, take_up, rng)
d0, d1 = 100 * (P0 - draws[:, 0]), 100 * (P1 - draws[:, 1])
print(f"baseline headcount P0 = {100 * P0:.2f}%, poverty gap P1 = {100 * P1:.2f}%")
print(f"budget per poor person (if perfectly targeted): "
      f"{budget / np.sum(w * (cons < zline)):.1f}")
for lab, d in (("headcount", d0), ("poverty gap", d1)):
    lo, hi = np.percentile(d, [5, 95])
    print(f"reduction in {lab:11s}: mean {d.mean():.3f} pp (MC s.e. "
          f"{d.std(ddof=1) / np.sqrt(S):.3f}), 90% band [{lo:.3f}, {hi:.3f}]")
# Benchmark: perfect targeting of the poorest 25%, full take-up, mpc = 1
best = simulate(0.0, 1.0, 1.0, rng)
print(f"benchmark (no PMT error, full take-up, mpc 1): P0 {100 * best[0]:.2f}%"
      f", P1 {100 * best[1]:.2f}%")

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.hist(d0, bins=40, color="steelblue", alpha=0.75, label="headcount (P0)")
ax.hist(d1, bins=40, color="darkorange", alpha=0.75, label="poverty gap (P1)")
ax.set_xlabel("reduction in poverty index (percentage points)")
ax.set_ylabel("number of simulation draws")
ax.legend()
plt.savefig("ch10_microsim.pdf", bbox_inches="tight")
