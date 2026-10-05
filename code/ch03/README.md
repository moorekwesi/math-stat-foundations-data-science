# Chapter 3 -- Eigenvalues, the SVD, PCA and Conditioning

Run all scripts from the book root with `python run_all.py ch03`
(outputs go to `*.out`, figures to `../../figures/`). All data are simulated
or bundled with scikit-learn (`load_digits`); no download is needed.

| Script | Book section | What it demonstrates |
|---|---|---|
| `eig_basics.py` | 3.1 Eigenvalues and eigenvectors | Characteristic polynomial, trace/determinant identities, diagonalisation, matrix powers, a defective (Jordan) matrix |
| `markov_labour.py` | 3.1 Markov chains | Simulated formal/informal/unemployment transition matrix; stationary distribution as a left eigenvector, spectral gap, half-life, simulation of 2000 workers; figure `ch03_markov_convergence.pdf` |
| `spectral_quadform.py` | 3.2 Symmetric matrices | Spectral theorem with `eigh`, three PD tests (eigenvalues, minors, Cholesky), Mahalanobis vs Euclidean distance, chi-square outlier rate; figure `ch03_mahalanobis.pdf` |
| `power_iteration.py` | 3.3 Computing eigenvalues | Power iteration rate (l2/l1)^(2k), Rayleigh quotient, shifted inverse iteration, Rayleigh-quotient iteration (cubic); figure `ch03_power_convergence.pdf` |
| `pagerank.py` | 3.3 PageRank | Google matrix of a simulated inter-district trade network, power iteration vs `eig`, ranking vs in-flow; `eig` vs `eigh` speed and orthogonality |
| `svd_basics.py` | 3.4 The SVD | Thin SVD, link to eig(X'X) and eig(XX'), rank and condition number, Eckart-Young-Mirsky check in 2- and Frobenius norms |
| `svd_digits.py` | 3.4 Low-rank approximation | Rank-k compression of an 80 x 80 mosaic of `load_digits` images; storage and error; figures `ch03_svd_digits.pdf`, `ch03_svd_spectrum.pdf` |
| `pca_macro.py` | 3.5 PCA and factor models | PCA via SVD of a simulated 40-series macro panel, explained variance, effect of standardisation, real-time diffusion-index forecast vs AR(1); figure `ch03_scree_macro.pdf` |
| `pca_asset_index.py` | 3.5 Asset index | Filmer-Pritchett wealth index on a simulated DHS-style survey, loadings and score weights, targeting accuracy vs asset count, biplot; figure `ch03_asset_index.pdf` |
| `conditioning_hilbert.py` | 3.6 Conditioning | Attaining the perturbation bound, Hilbert matrices (error ~ cond * eps), normal equations vs QR for polynomial regression; figure `ch03_hilbert.pdf` |
| `vif_collinearity.py` | 3.6 Multicollinearity | VIFs from R^-1 and auxiliary regressions, eigenvalues and condition indices, Monte Carlo spread of OLS vs ridge; figure `ch03_vif_ridge.pdf` |
