# DEP 803 -- Chapter 1: Notation, Vectors, Matrices and Computational Precision
# Seeds, independent random streams, recorded versions and summation order.
# (c) 2026 Stephen E. Moore. MIT Licence.
import sys
import numpy as np
import scipy
import pandas as pd

# 1. the same seed gives the same draws
a = np.random.default_rng(803).normal(size=3)
b = np.random.default_rng(803).normal(size=3)
c = np.random.default_rng(804).normal(size=3)
print("seed 803:", a.round(6))
print("seed 803 again identical:", np.array_equal(a, b),
      "  seed 804 differs:", not np.array_equal(a, c))

# 2. independent streams for parallel workers (e.g. bootstrap replications)
ss = np.random.SeedSequence(803)
streams = [np.random.default_rng(s) for s in ss.spawn(3)]
print("first draw of 3 spawned streams:",
      [round(g.uniform(), 6) for g in streams])

# 3. record the computational environment alongside results
print("python", sys.version.split()[0], "| numpy", np.__version__,
      "| scipy", scipy.__version__, "| pandas", pd.__version__)

# 4. floating-point addition is not associative: order changes the answer
rng = np.random.default_rng(803)
x = rng.lognormal(10, 2, 1_000_000)                  # skewed "incomes"
s1 = np.sum(x)
s2 = np.sum(rng.permutation(x))
s3 = np.sum(np.sort(x))
print("total, original order : %.6f" % s1)
print("total, shuffled order : %.6f" % s2)
print("total, sorted order   : %.6f" % s3)
print("relative differences  : %.1e, %.1e" % ((s2 - s1) / s1, (s3 - s1) / s1))
print("(0.1 + 0.2) + 0.3 == 0.1 + (0.2 + 0.3) ?",
      (0.1 + 0.2) + 0.3 == 0.1 + (0.2 + 0.3))
