# DEP 803 -- Chapter 10: Numerical Integration and Monte Carlo Simulation
# Stratified sampling: an integral, and a simulated three-region survey.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)

# (a) theta = int_0^1 exp(u) du with K equal-width strata, 1 draw per stratum
n, R = 1000, 2000
plain = np.exp(rng.random((R, n))).mean(axis=1)
k = np.arange(n)
strat = np.exp((k + rng.random((R, n))) / n).mean(axis=1)
print(f"integral of exp on [0,1], n = {n}, {R} replications")
print(f"   plain      sd {plain.std():.2e}")
print(f"   stratified sd {strat.std():.2e}   (variance ratio "
      f"{plain.var() / strat.var():.0f})")

# (b) A simulated population of 60,000 households in three regions
N_h = np.array([30_000, 20_000, 10_000])               # stratum sizes
mu_h, sd_h = np.array([7.4, 6.9, 6.5]), np.array([0.5, 0.6, 0.9])
pop = [np.exp(rng.normal(mu_h[h], sd_h[h], N_h[h])) for h in range(3)]
W = N_h / N_h.sum()
zline = 700.0                                           # poverty line
poor = [(p < zline).astype(float) for p in pop]
true = sum(W[h] * poor[h].mean() for h in range(3))
S_h = np.array([poor[h].std() for h in range(3)])
allpop = np.concatenate(poor)

n = 600
n_prop = np.round(n * W).astype(int)
n_ney = np.round(n * W * S_h / (W * S_h).sum()).astype(int)
est = {"SRS": [], "proportional": [], "Neyman": []}
for _ in range(R):
    est["SRS"].append(rng.choice(allpop, n, replace=False).mean())
    for lab, nh in (("proportional", n_prop), ("Neyman", n_ney)):
        e = sum(W[h] * rng.choice(poor[h], nh[h], replace=False).mean()
                for h in range(3))
        est[lab].append(e)
print(f"\npoverty headcount, true = {true:.4f}; n = {n}; {R} surveys")
print(f"   stratum headcounts {[round(p.mean(), 3) for p in poor]}")
print(f"   Neyman allocation  {n_ney.tolist()}  (proportional "
      f"{n_prop.tolist()})")
v0 = np.var(est["SRS"])
for lab, e in est.items():
    e = np.asarray(e)
    print(f"   {lab:13s} mean {e.mean():.4f}  sd {e.std():.4f}  "
          f"design effect {e.var() / v0:.3f}")
