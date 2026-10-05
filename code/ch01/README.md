# Chapter 1 -- Mathematical Notation, Vectors, Matrices and Computational Precision

Run all scripts from the book root with `python run_all.py ch01` (figures are written to
`figures/`, printed output to `code/ch01/*.out`). All data are simulated or illustrative.

| Script | Book section | What it demonstrates |
|---|---|---|
| `notation_basics.py` | 1.1 | Index, summation and product notation in NumPy; empty sums/products; underflow of a product of probabilities |
| `bigO_growth.py` | 1.1 | Growth of log n, n, n log n, n^2, n^3 (figure `ch01_growth.pdf`) and run-time estimates |
| `vectors_norms.py` | 1.2 | Inner products, l1/l2/l-inf norms, angles, Cauchy-Schwarz, triangle inequality, norm equivalence, cosine similarity |
| `unit_balls.py` | 1.2 | Unit balls of the l1, l2, l-inf norms (figure `ch01_unitballs.pdf`) |
| `cpi_inner_product.py` | 1.2 | Laspeyres, Paasche and Fisher price indices as inner products; item contributions |
| `matrix_products.py` | 1.3 | Four views of the matrix product; transpose/inverse rules; orthogonal and idempotent matrices |
| `block_inverse.py` | 1.3 | Partitioned inverse via the Schur complement; trace and determinant identities; `slogdet` |
| `det_geometry.py` | 1.3 | Determinant as area-scaling factor (figure `ch01_determinant.pdf`) |
| `broadcasting.py` | 1.3 | NumPy broadcasting: centring, row scaling, pairwise distances, the (n,) vs (n,1) trap |
| `leontief.py` | 1.3 | Three-sector Leontief input-output model, multipliers, Neumann series |
| `data_matrix.py` | 1.4 | pandas to NumPy, centring matrix H, sample covariance and correlation |
| `standardise_fig.py` | 1.4 | Raw versus standardised survey data (figure `ch01_standardise.pdf`) |
| `kronecker.py` | 1.4 | Kronecker product rules, vec identity, panel-data dummy matrices |
| `float_basics.py` | 1.5 | IEEE 754 doubles: epsilon, spacing, overflow, underflow, NaN, integer wrap-around, `np.isclose` |
| `cancellation.py` | 1.5 | Catastrophic cancellation and stable rewrites (`expm1`, half-angle, stable quadratic roots) |
| `cancellation_fig.py` | 1.5 | Relative error of (exp(x)-1)/x versus eps/x (figure `ch01_cancellation.pdf`) |
| `logsumexp.py` | 1.5 | Stable softmax, log-sum-exp, binary logit log-likelihood via `logaddexp` |
| `summation.py` | 1.5 | Naive, Kahan, pairwise and `math.fsum` summation in float64 and float32 |
| `vancouver.py` | 1.5 | Truncation versus rounding in a frequently updated index (figure `ch01_vancouver.pdf`) |
| `variance_algorithms.py` | 1.5 | One-pass, two-pass and Welford variance under a large offset; float32 |
| `variance_fig.py` | 1.5 | Variance error against offset (figure `ch01_variance.pdf`) |
| `rounding_money.py` | 1.5 | Round-half-even, `decimal`, VAT rounding per line vs on total, rounded published components |
| `timing_vectorise.py` | 1.6 | Loops versus vectorised code; cubic cost of matrix products and solves (figure `ch01_timing.pdf`) |
| `sparse_admin.py` | 1.6 | CSR sparse matrices for a simulated employer-employee register |
| `reproducibility.py` | 1.6 | Seeds, `SeedSequence.spawn`, recording versions, order-dependence of floating-point sums |
