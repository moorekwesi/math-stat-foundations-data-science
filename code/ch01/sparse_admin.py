# DEP 803 -- Chapter 1: Notation, Vectors, Matrices and Computational Precision
# Sparse matrices for a simulated employer-employee administrative dataset.
# (c) 2026 Stephen E. Moore. MIT Licence.
import time
import numpy as np
import scipy.sparse as sp

rng = np.random.default_rng(803)
n_workers, n_firms, n_dist = 500_000, 20_000, 250
firm = rng.integers(0, n_firms, n_workers)            # employer of each worker
district = rng.integers(0, n_dist, n_firms)[firm]     # firm location
wage = rng.lognormal(7.0, 0.6, n_workers)             # simulated monthly wage

# incidence matrix F: F[i, j] = 1 if worker i is employed by firm j
F = sp.csr_matrix((np.ones(n_workers), (np.arange(n_workers), firm)),
                  shape=(n_workers, n_firms))
dense_gb = n_workers * n_firms * 8 / 1e9
sparse_mb = (F.data.nbytes + F.indices.nbytes + F.indptr.nbytes) / 1e6
print(f"F is {F.shape[0]:,} x {F.shape[1]:,} with {F.nnz:,} non-zeros"
      f" (density {F.nnz / np.prod(F.shape):.1e})")
print(f"dense storage would need {dense_gb:,.0f} GB; CSR uses {sparse_mb:.1f} MB")

t0 = time.perf_counter()
size = F.T @ np.ones(n_workers)                       # workers per firm
mean_wage = (F.T @ wage) / np.maximum(size, 1)        # average wage per firm
t1 = time.perf_counter()
print(f"firm sizes and mean wages via F^T x in {1e3 * (t1 - t0):.1f} ms")
print(f"largest firm: {int(size.max())} workers; mean firm size {size.mean():.1f}")
check = np.bincount(firm, weights=wage, minlength=n_firms) / np.maximum(size, 1)
print("agrees with np.bincount:", np.allclose(mean_wage, check))

# district dummy design matrix and its cross-product
D = sp.csr_matrix((np.ones(n_workers), (np.arange(n_workers), district)),
                  shape=(n_workers, n_dist))
DtD = (D.T @ D).toarray()
print("D^T D is diagonal:", np.count_nonzero(DtD - np.diag(np.diag(DtD))) == 0,
      "  diagonal = workers per district, e.g.", np.diag(DtD)[:4].astype(int))
coef = (D.T @ np.log(wage)) / np.diag(DtD)            # OLS on dummies = means
print("OLS on district dummies = district means of log wage:",
      np.allclose(coef[0], np.log(wage)[district == 0].mean()))

# worker-to-worker co-employment network among a 2,000-worker sample
S = F[:2000]
Co = S @ S.T                                          # co-worker indicator
print(f"co-employment matrix (2000 x 2000): {Co.nnz:,} non-zeros,"
      f" format {Co.format}")
