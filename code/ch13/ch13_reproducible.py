# DEP 803 -- Chapter 13: Problem Set Clinic and Applied Foundations Review
# Reproducibility toolkit: seeds, independent streams, environment, fingerprints.
# (c) 2026 Stephen E. Moore. MIT Licence.
import hashlib
import platform
import numpy as np
import scipy
import sklearn


def experiment(rng, n=1000):
    """A toy 'analysis': bootstrap s.e. of a mean of simulated incomes."""
    income = rng.lognormal(mean=6.0, sigma=0.8, size=n)
    idx = rng.integers(0, n, size=(500, n))
    return income.mean(), income[idx].mean(axis=1).std(ddof=1)


def fingerprint(*arrays):
    """Short SHA-256 hash of results: equal hashes => bitwise equal results."""
    h = hashlib.sha256()
    for a in arrays:
        h.update(np.ascontiguousarray(a, dtype=float).tobytes())
    return h.hexdigest()[:16]


# 1. Same seed => identical results; no seed => different results
r1 = experiment(np.random.default_rng(803))
r2 = experiment(np.random.default_rng(803))
r3 = experiment(np.random.default_rng())
print("seed 803, run 1 : mean %.3f  boot se %.4f  hash %s"
      % (*r1, fingerprint(r1)))
print("seed 803, run 2 : mean %.3f  boot se %.4f  hash %s"
      % (*r2, fingerprint(r2)))
print("no seed         : identical to run 1?", fingerprint(r3) == fingerprint(r1))

# 2. Independent streams for parallel workers via SeedSequence.spawn
root = np.random.SeedSequence(803)
streams = [np.random.default_rng(s) for s in root.spawn(4)]
first = np.array([g.standard_normal() for g in streams])
print("\nfirst draw of 4 spawned streams:", np.array2string(first, precision=4))
print("spawned stream 0 reproducible?  ",
      np.random.default_rng(np.random.SeedSequence(803).spawn(4)[0])
      .standard_normal() == first[0])

# 3. Record the computational environment alongside every result
print("\nenvironment:")
print(f"  python {platform.python_version()}, numpy {np.__version__}, "
      f"scipy {scipy.__version__}, sklearn {sklearn.__version__}")
