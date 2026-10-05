# Chapter 2 -- Linear Algebra for Modelling: Rank, Projections and Decompositions

All data are simulated (fixed seed 803). Run from the book root with
`python run_all.py ch02`; figures are written to `figures/`.

| Script | Book section | What it demonstrates |
|---|---|---|
| `subspaces.py` | 2.1 Linear systems and vector spaces | Four fundamental subspaces from the SVD, dimension counts, orthogonality, solvability of `Ax = b` via ranks |
| `dummy_trap.py` | 2.2 Rank and identification | Dummy variable trap with 16 regions, rank-nullity check, numerical rank and the tolerance in `np.linalg.matrix_rank` |
| `identification.py` | 2.2 Rank and identification | Programme effect collinear with region effects, estimable functions (row space), identification restored by a second survey round (DiD) |
| `projection.py` | 2.3 Orthogonality and projections | Hat matrix `P` and annihilator `M`: symmetry, idempotency, trace = rank, Pythagoras, ANOVA and R^2 |
| `gram_schmidt.py` | 2.3 Orthogonality and projections | Classical vs modified Gram-Schmidt vs Householder: loss of orthogonality (figure `ch02_gram_schmidt.pdf`) |
| `fwl_panel.py` | 2.4 Least squares as projection | Frisch-Waugh-Lovell: LSDV with 400 worker dummies equals the within (demeaned) estimator in a simulated wage panel (figure `ch02_fwl_panel.pdf`) |
| `leverage.py` | 2.4 Least squares as projection | Leverages from a thin QR, sum = k, high-leverage household, leave-one-out residual formula |
| `lu_cholesky.py` | 2.5 Matrix decompositions | LU factor-and-solve for a Leontief input-output model; Cholesky simulation of correlated variables and positive-definiteness test |
| `qr_vs_normal.py` | 2.5 Matrix decompositions | Householder QR (own code) vs normal equations vs SVD on ill-conditioned polynomial regression (figure `ch02_qr_vs_normal.pdf`) |
| `pinv_penrose.py` | 2.6 Generalised inverses | Four Penrose conditions, minimum-norm least squares, `pinv` vs `lstsq`, interpolation when p > n, ridge limit |
| `min_norm_descent.py` | 2.6 Generalised inverses | Test error of minimum-norm least squares across p = n (double descent; figure `ch02_min_norm_descent.pdf`) |
| `balancing.py` | 2.6 Generalised inverses | Reliability-weighted balancing of illustrative national accounts with redundant constraints using `pinv` |
