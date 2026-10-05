# DEP 803 -- Chapter 8: Optimisation
# Minibatch SGD for a logit on a large simulated administrative register
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.default_rng(803)
n, p = 200_000, 10
X = np.column_stack([np.ones(n), rng.normal(size=(n, p - 1))])
beta_true = np.concatenate([[-1.0], rng.normal(scale=0.5, size=p - 1)])
y = (rng.uniform(size=n) < 1 / (1 + np.exp(-X @ beta_true))).astype(float)


def loss(b):
    z = X @ b
    return np.mean(np.logaddexp(0, z) - y * z)


def minibatch_grad(b, idx):
    Xb = X[idx]
    return Xb.T @ (1 / (1 + np.exp(-Xb @ b)) - y[idx]) / idx.size


# reference optimum by Newton's method (10 iterations are plenty)
b_star = np.zeros(p)
for _ in range(10):
    pr = 1 / (1 + np.exp(-X @ b_star))
    H = (X * (pr * (1 - pr))[:, None]).T @ X / n
    b_star -= np.linalg.solve(H, X.T @ (pr - y) / n)
f_star = loss(b_star)


def sgd(schedule, batch=64, epochs=3, seed=1):
    r = np.random.default_rng(seed)
    b, k, curve = np.zeros(p), 0, []
    steps_per_epoch = n // batch
    for ep in range(epochs):
        perm = r.permutation(n)                  # reshuffle every epoch
        for s in range(steps_per_epoch):
            idx = perm[s * batch:(s + 1) * batch]
            b -= schedule(k) * minibatch_grad(b, idx)
            k += 1
            if k % 250 == 0:
                curve.append(loss(b) - f_star)
    return b, np.array(curve)


schedules = {
    "constant 0.5": lambda k: 0.5,
    "constant 0.05": lambda k: 0.05,
    "1/sqrt(k) decay": lambda k: 0.5 / np.sqrt(1 + k / 100),
    "step decay": lambda k: 0.5 * 0.1 ** (k // 3125),
}
print(f"n = {n:,}, batch 64, 3 epochs = {3 * n // 64:,} updates")
print(f"{'schedule':<17}{'excess loss':>13}{'||b - b*||':>12}")
plt.figure(figsize=(6, 3.8))
for name, sch in schedules.items():
    b, curve = sgd(sch)
    print(f"{name:<17}{loss(b) - f_star:>13.2e}"
          f"{np.linalg.norm(b - b_star):>12.4f}")
    plt.semilogy(250 * np.arange(1, curve.size + 1), np.abs(curve) + 1e-9,
                 label=name, lw=1)
for batch in [8, 512]:
    b, _ = sgd(schedules["constant 0.5"], batch=batch)
    print(f"constant 0.5, batch {batch:>3}: excess loss "
          f"{loss(b) - f_star:.2e}")
plt.gca().set(xlabel="SGD update k", ylabel="excess loss  f(b_k) - f*",
              title="Learning-rate schedules and the SGD noise floor")
plt.legend(frameon=False)
plt.savefig("ch08_sgd_schedules.pdf", bbox_inches="tight")
