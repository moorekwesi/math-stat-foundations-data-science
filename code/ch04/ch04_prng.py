# DEP 803 -- Chapter 4: Probability Foundations, Random Variables and Transformations
# Pseudo-random number generators: middle-square, RANDU, and NumPy's PCG64.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt


def middle_square(x, steps):
    """von Neumann: square a 4-digit number and keep the middle 4 digits."""
    out = [x]
    for _ in range(steps):
        x = (x * x // 100) % 10_000
        out.append(x)
    return out


seq = middle_square(1234, 60)
first_repeat = next(i for i, v in enumerate(seq) if v in seq[:i])
print("Middle-square from 1234:", " ".join(f"{v:04d}" for v in seq[:10]), "...")
print(f"  value {seq[first_repeat]:04d} recurs at step {first_repeat}: "
      "the sequence has fallen into a short cycle")

# RANDU: x_{k+1} = 65539 x_k mod 2^31  (widely used in the 1960s-70s)
m = 2**31
x = np.empty(30_002, dtype=np.int64)
x[0] = 1
for k in range(1, x.size):
    x[k] = (65539 * x[k - 1]) % m
lhs = (x[2:] - 6 * x[1:-1] + 9 * x[:-2]) % m
print(f"\nRANDU: x[k+2] - 6 x[k+1] + 9 x[k] = 0 (mod 2^31) for all k? "
      f"{bool(np.all(lhs == 0))}")
u = x / m
c_randu = 9 * u[:-2] - 6 * u[1:-1] + u[2:]

# Modern practice: one seeded Generator (PCG64), independent child streams
rng = np.random.default_rng(803)
print(f"\nbit generator: {type(rng.bit_generator).__name__}")
a = np.random.default_rng(803).random(3)
b = np.random.default_rng(803).random(3)
print("same seed, same stream:", np.round(a, 6), np.array_equal(a, b))
children = np.random.SeedSequence(803).spawn(3)
streams = [np.random.default_rng(s) for s in children]
print("spawned streams, first draw each:",
      np.round([g.random() for g in streams], 6))
v = rng.random(30_002)
c_pcg = 9 * v[:-2] - 6 * v[1:-1] + v[2:]

fig, axes = plt.subplots(1, 2, figsize=(7.5, 3.2), sharey=True)
for a_, c, title in [(axes[0], c_randu, "RANDU"), (axes[1], c_pcg, "PCG64")]:
    a_.hist(c, bins=400, range=(-6, 10), density=True)
    a_.set_title(title)
    a_.set_xlabel("9 u[k] - 6 u[k+1] + u[k+2]")
    a_.grid(alpha=0.3)
axes[0].set_ylabel("density")
fig.tight_layout()
plt.savefig("ch04_prng.pdf", bbox_inches="tight")
