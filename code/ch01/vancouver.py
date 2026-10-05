# DEP 803 -- Chapter 1: Notation, Vectors, Matrices and Computational Precision
# A stylised re-enactment of the Vancouver index truncation error (simulated).
# (c) 2026 Stephen E. Moore. MIT Licence.
import math
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
rng = np.random.default_rng(803)
days, per_day = 440, 2800                 # about 22 months of trading
m = days * per_day
r = rng.normal(0.0, 2e-5, m)              # tiny zero-mean index changes per trade
growth = 1.0 + r

exact = 1000.0 * np.cumprod(growth)       # full-precision index
trunc = np.empty(m)
rounded = np.empty(m)
t = u = 1000.0
for k, g in enumerate(growth.tolist()):   # recompute after every trade
    t = math.floor(t * g * 1000) / 1000   # truncate to 3 decimals
    u = round(u * g, 3)                   # round to 3 decimals
    trunc[k], rounded[k] = t, u

print(f"index updates simulated       : {m:,}")
print(f"final index, full precision   : {exact[-1]:10.3f}")
print(f"final index, rounded (3 dp)   : {rounded[-1]:10.3f}")
print(f"final index, truncated (3 dp) : {trunc[-1]:10.3f}")
print(f"truncation drift              : {trunc[-1] - exact[-1]:10.3f}")
print(f"predicted drift ~ -0.0005 x m : {-0.0005 * m:10.1f}")

day_end = np.arange(per_day - 1, m, per_day)
fig, ax = plt.subplots(figsize=(6, 3.8))
ax.plot(exact[day_end], lw=1.6, label="full precision")
ax.plot(rounded[day_end], lw=1.0, ls="--", label="rounded to 3 decimals")
ax.plot(trunc[day_end], lw=1.6, label="truncated to 3 decimals")
ax.set_xlabel("trading day")
ax.set_ylabel("index level")
ax.set_title("Simulated index: truncation drifts, rounding does not")
ax.legend(frameon=False)
plt.savefig("ch01_vancouver.pdf", bbox_inches="tight")
