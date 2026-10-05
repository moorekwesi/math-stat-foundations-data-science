# DEP 803 -- Chapter 4: Probability Foundations, Random Variables and Transformations
# Sums of independent variables: numerical convolution versus simulation.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

rng = np.random.default_rng(803)
h = 0.01
x = np.arange(0, 30, h)
f_exp = stats.expon(scale=2.0).pdf(x)        # waiting time, mean 2

# Density of S_k = sum of k iid Exp(mean 2) by repeated discrete convolution
dens = {1: f_exp}
for k in range(2, 6):
    dens[k] = np.convolve(dens[k - 1], f_exp)[: x.size] * h
print("Sum of k iid exponentials (mean 2) is Gamma(k, scale 2)")
print(f"{'k':>2} {'max|conv - gamma|':>18} {'P(S>10) conv':>13} {'P(S>10) sim':>12}")
for k in range(1, 6):
    err = np.max(abs(dens[k] - stats.gamma(k, scale=2.0).pdf(x)))
    tail = dens[k][x > 10].sum() * h
    sim = np.mean(rng.exponential(2.0, (200_000, k)).sum(1) > 10)
    print(f"{k:>2} {err:18.4f} {tail:13.4f} {sim:12.4f}")

# Discrete convolution: total of two dice
die = np.full(6, 1 / 6)
two = np.convolve(die, die)
print("\nPMF of the sum of two dice (values 2..12):")
print("  " + " ".join(f"{36 * p:.0f}" for p in two) + "   (times 1/36)")

fig, ax = plt.subplots(figsize=(6, 3.8))
for k in [1, 2, 3, 5]:
    ax.plot(x, dens[k], lw=1.5, label=f"k = {k}")
ax.set_xlim(0, 25)
ax.set(xlabel="s", ylabel="density of S_k")
ax.set_title("convolution powers of an exponential density")
ax.legend(frameon=False)
ax.grid(alpha=0.3)
plt.savefig("ch04_convolution.pdf", bbox_inches="tight")
