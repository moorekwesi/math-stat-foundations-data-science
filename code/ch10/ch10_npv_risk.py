# DEP 803 -- Chapter 10: Numerical Integration and Monte Carlo Simulation
# Monte Carlo risk assessment of a (hypothetical) public irrigation investment.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
R, T = 100_000, 25                      # replications, project life (years)
# Uncertain inputs (illustrative assumptions, monetary units = millions)
cost = rng.triangular(80, 100, 140, R)           # capital cost, overrun risk
delay = rng.integers(0, 3, R)                    # years before benefits start
b0 = rng.lognormal(np.log(9.0), 0.25, R)         # first-year net benefit
growth = rng.normal(0.02, 0.01, R)               # annual benefit growth
rate = 0.06                                      # social discount rate

t = np.arange(1, T + 1)
active = t[None, :] > delay[:, None]
flows = b0[:, None] * (1 + growth[:, None]) ** (t[None, :] - 1) * active
npv = (flows / (1 + rate) ** t).sum(axis=1) - cost

m, se = npv.mean(), npv.std(ddof=1) / np.sqrt(R)
p_loss = np.mean(npv < 0)
se_p = np.sqrt(p_loss * (1 - p_loss) / R)
q05, q50, q95 = np.percentile(npv, [5, 50, 95])
print(f"expected NPV        : {m:8.2f}  (MC s.e. {se:.2f})")
print(f"P(NPV < 0)          : {p_loss:8.4f}  (MC s.e. {se_p:.4f})")
print(f"5%, 50%, 95% NPV    : {q05:8.2f} {q50:8.2f} {q95:8.2f}")
# Which input matters most? rank correlation of each input with NPV
ranks = lambda z: np.argsort(np.argsort(z))
for name, z in [("cost", cost), ("delay", delay + rng.random(R) * 1e-6),
                ("benefit", b0), ("growth", growth)]:
    rho = np.corrcoef(ranks(z), ranks(npv))[0, 1]
    print(f"   Spearman corr(NPV, {name:8s}) = {rho: .3f}")

fig, ax = plt.subplots(figsize=(6, 3.8))
ax.hist(npv, bins=120, density=True, color="steelblue", alpha=0.8)
ax.axvline(0, color="firebrick", lw=1.5, label=f"P(NPV<0) = {p_loss:.3f}")
ax.axvline(m, color="black", ls="--", lw=1, label=f"E[NPV] = {m:.1f}")
ax.set_xlabel("net present value (millions, simulated)")
ax.set_ylabel("density")
ax.legend()
plt.savefig("ch10_npv_risk.pdf", bbox_inches="tight")
