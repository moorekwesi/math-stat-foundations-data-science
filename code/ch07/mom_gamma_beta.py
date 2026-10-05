# DEP 803 -- Chapter 7: Estimation Principles
# Method-of-moments estimators for the gamma and beta families.
# (c) 2026 Stephen E. Moore. MIT Licence.
import numpy as np

rng = np.random.default_rng(803)

def mom_gamma(x):
    m, v = x.mean(), x.var()
    return m**2 / v, v / m                     # shape k, scale theta

def mom_beta(x):
    m, v = x.mean(), x.var()
    c = m * (1 - m) / v - 1
    return m * c, (1 - m) * c                  # a, b

# Simulated household consumption (gamma) and budget shares (beta)
k0, th0, a0, b0, n, R = 2.5, 1.6, 2.0, 5.0, 400, 5000
xg = rng.gamma(k0, th0, size=n)
xb = rng.beta(a0, b0, size=n)
print(f"One sample of n = {n}")
print("  gamma: true (k, theta) = (%.2f, %.2f), MoM = (%.3f, %.3f)"
      % ((k0, th0) + mom_gamma(xg)))
print("  beta : true (a, b)     = (%.2f, %.2f), MoM = (%.3f, %.3f)"
      % ((a0, b0) + mom_beta(xb)))

# Sampling properties over R replications
G = rng.gamma(k0, th0, size=(R, n))
B = rng.beta(a0, b0, size=(R, n))
mg, vg = G.mean(axis=1), G.var(axis=1)
mb, vb = B.mean(axis=1), B.var(axis=1)
k_hat, th_hat = mg**2 / vg, vg / mg
c = mb * (1 - mb) / vb - 1
a_hat, b_hat = mb * c, (1 - mb) * c
print(f"\nMonte Carlo over R = {R} samples of size n = {n}")
print(f"{'param':>6} {'true':>6} {'mean est':>9} {'bias':>8} {'sd':>7}")
for name, t, e in (("k", k0, k_hat), ("theta", th0, th_hat),
                   ("a", a0, a_hat), ("b", b0, b_hat)):
    print(f"{name:>6} {t:6.2f} {e.mean():9.3f} {e.mean() - t:8.3f}"
          f" {e.std():7.3f}")
