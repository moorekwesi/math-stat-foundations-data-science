# DEP 803 -- Chapter 7: Estimation Principles
# Sampling distributions of OLS and IV under endogeneity, strong vs weak instrument.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
R, n, beta = 5000, 500, 0.08

def simulate(pi):
    """Return OLS and IV slope estimates for R samples; pi = first-stage slope."""
    a = rng.normal(size=(R, n))
    z = rng.normal(size=(R, n))
    s = pi * z + a + rng.normal(size=(R, n))
    y = beta * s + 0.25 * a + 0.5 * rng.normal(size=(R, n))
    sc = s - s.mean(axis=1, keepdims=True)
    yc = y - y.mean(axis=1, keepdims=True)
    zc = z - z.mean(axis=1, keepdims=True)
    ols = (sc * yc).sum(1) / (sc * sc).sum(1)
    iv = (zc * yc).sum(1) / (zc * sc).sum(1)
    return ols, iv

print(f"R = {R} samples of n = {n}; true beta = {beta}")
print(f"{'design':>8} {'estimator':>9} {'median':>8} {'IQR':>8}")
res = {}
for label, pi in (("strong", 1.0), ("weak", 0.08)):
    ols, iv = simulate(pi)
    res[label] = (ols, iv)
    for name, e in (("OLS", ols), ("IV", iv)):
        q1, q2, q3 = np.percentile(e, [25, 50, 75])
        print(f"{label:>8} {name:>9} {q2:8.4f} {q3 - q1:8.4f}")

plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(6, 3.8))
bins = np.linspace(-0.1, 0.3, 120)
ax.hist(res["strong"][0], bins, density=True, alpha=0.6, label="OLS")
ax.hist(res["strong"][1], bins, density=True, alpha=0.6,
        label="IV, strong instrument")
wk = res["weak"][1]                      # many draws fall outside the window
ax.hist(wk, bins, weights=np.full(R, 1 / (R * (bins[1] - bins[0]))),
        histtype="step", lw=1.4, color="k", label="IV, weak instrument")
ax.axvline(beta, color="tab:red", ls="--", lw=1, label="true beta")
ax.set_xlabel("estimated return to schooling")
ax.set_ylabel("density")
ax.legend(frameon=False, fontsize=9)
plt.savefig("ch07_iv_montecarlo.pdf", bbox_inches="tight")
