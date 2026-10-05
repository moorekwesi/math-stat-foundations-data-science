# DEP 803 -- Chapter 4: Probability Foundations, Random Variables and Transformations
# Proxy means test (PMT) targeting in a simulated population: Bayes in action.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
N = 200_000                                  # simulated households
log_c = rng.normal(0.0, 0.6, N)             # log consumption per adult (simulated)
z = np.quantile(log_c, 0.25)                # poverty line: 25% are poor
poor = log_c < z
pmt = log_c + rng.normal(0.0, 0.35, N)      # PMT score = noisy prediction
cut = np.quantile(pmt, 0.20)                # programme covers 20% of households
selected = pmt < cut

sens = selected[poor].mean()                 # P(selected | poor)
spec = (~selected[~poor]).mean()             # P(not selected | non-poor)
ppv = poor[selected].mean()                  # P(poor | selected)
prev = poor.mean()
ppv_bayes = sens * prev / (sens * prev + (1 - spec) * (1 - prev))
print(f"Prevalence of poverty  P(poor)              = {prev:.3f}")
print(f"Coverage               P(sel)               = {selected.mean():.3f}")
print(f"Sensitivity            P(sel | poor)        = {sens:.3f}")
print(f"Specificity            P(not sel | nonpoor) = {spec:.3f}")
print(f"P(poor | sel), direct count                 = {ppv:.3f}")
print(f"P(poor | sel), Bayes' theorem               = {ppv_bayes:.3f}")
print(f"Inclusion error  = 1 - P(poor | sel)        = {1 - ppv:.3f}")
print(f"Exclusion error  = 1 - P(sel | poor)        = {1 - sens:.3f}")

print("\nSame PMT accuracy, different coverage of the programme:")
print(f"{'coverage':>9} {'inclusion':>10} {'exclusion':>10}")
for cov in [0.10, 0.20, 0.25, 0.30, 0.40]:
    s = pmt < np.quantile(pmt, cov)
    print(f"{cov:9.2f} {1 - poor[s].mean():10.3f} {1 - s[poor].mean():10.3f}")

fig, ax = plt.subplots(figsize=(6, 3.8))
bins = np.linspace(-2.5, 2.5, 80)
ax.hist(pmt[poor], bins, density=True, alpha=0.6, label="truly poor")
ax.hist(pmt[~poor], bins, density=True, alpha=0.6, label="truly non-poor")
ax.axvline(cut, color="k", ls="--", lw=1.2, label="eligibility cut-off")
ax.set(xlabel="PMT score (predicted log consumption)", ylabel="density")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
plt.savefig("ch04_targeting.pdf", bbox_inches="tight")
