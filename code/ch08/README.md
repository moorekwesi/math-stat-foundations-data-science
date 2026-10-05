# Chapter 8 -- Optimisation: Convexity, Gradients, Hessians, Newton and Stochastic Gradient Methods

All data are simulated (seeded with `np.random.default_rng(803)` unless noted).
Run from the book root with `python run_all.py ch08`; figures go to `figures/`.
On heavily loaded multi-core machines, setting `OPENBLAS_NUM_THREADS=1` can make the
matrix-heavy scripts much faster. Timings printed by `ch08_bfgs.py` are machine dependent.

| Script | Book section | What it demonstrates |
|---|---|---|
| `ch08_mestimators.py` | 8.1 | OLS, LAD and Huber regression as numerical M-estimators on contaminated earnings data; figure `ch08_losses.pdf` |
| `ch08_weierstrass.py` | 8.1 | Local vs global minima; multi-start BFGS on a non-convex function vs grid search |
| `ch08_critical.py` | 8.2 | Solving grad f = 0 from many starts and classifying stationary points by Hessian eigenvalues; figure `ch08_critical.pdf` |
| `ch08_gradcheck.py` | 8.2 | Forward/central finite differences, gradient checking of the logit loss, optimal step size; figure `ch08_fd_error.pdf` |
| `ch08_convexity.py` | 8.3 | Logit Hessian eigenvalues, ridge constants mu, L and kappa, effect of standardising, Jensen check, Cobb-Douglas concavity vs quasi-concavity |
| `ch08_gd_quadratic.py` | 8.4 | GD iteration counts vs condition number (theory vs practice); zig-zag vs heavy ball; figure `ch08_zigzag.pdf` |
| `ch08_gd_compare.py` | 8.4 | GD, Armijo backtracking, heavy ball and Nesterov on unscaled least squares; feature scaling; figure `ch08_gd_rates.pdf` |
| `ch08_newton_damping.py` | 8.5 | Divergence of pure Newton on sqrt(1+x^2) and the damped (backtracking) fix |
| `ch08_newton_logit.py` | 8.5 | Newton-Raphson / IRLS for a logit of programme take-up, quadratic convergence, standard errors, comparison with sklearn and GD; figure `ch08_newton_conv.pdf` |
| `ch08_gauss_newton.py` | 8.5 | Gauss-Newton vs Levenberg-Marquardt for a logistic diffusion (adoption) curve; figure `ch08_gauss_newton.pdf` |
| `ch08_bfgs.py` | 8.5 | `scipy.optimize.minimize` methods (Nelder-Mead, CG, BFGS, L-BFGS-B, Newton-CG, trust-exact) on Rosenbrock and a logit |
| `ch08_sgd.py` | 8.6 | Minibatch SGD for a logit on a 200,000-record simulated register; learning-rate schedules, noise floor, batch size; figure `ch08_sgd_schedules.pdf` |
| `ch08_adam.py` | 8.6 | SGD, momentum, AdaGrad, RMSProp and Adam implemented in NumPy on a badly scaled logit; sklearn `SGDClassifier` with and without scaling; figure `ch08_adam.pdf` |
