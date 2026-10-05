# Chapter 12 -- Matrix Calculus for Machine Learning, Regularisation and the Bias-Variance Trade-off

Run all scripts from the book root with `python run_all.py ch12` (figures go to `figures/`).
All data are simulated or bundled with scikit-learn (`load_digits`, `load_diabetes`).

| Script | Book section | What it demonstrates |
|---|---|---|
| `matcalc_fd_check.py` | 12.1 Matrix calculus | Central-difference check of gradients of linear/quadratic forms, traces, log det and inverses; Hessian of a quadratic form |
| `fd_step_jacobian.py` | 12.1 Matrix calculus | Softmax Jacobian, chain-rule Jacobian, forward vs central difference error as a function of h (`ch12_fd_error.pdf`) |
| `ols_mvn_mle.py` | 12.2 Applications | OLS from the gradient and Hessian; Gaussian MLE of mean and covariance with analytic and numerical score |
| `logit_newton.py` | 12.2 Applications | Logistic regression gradient/Hessian, Newton-Raphson (IRLS), comparison with scikit-learn |
| `softmax_gls.py` | 12.2 Applications | Multinomial logit gradient X'(P - Y); GLS vs OLS sampling variance under heteroskedasticity |
| `autodiff_tiny.py` | 12.3 Automatic differentiation | Forward mode (dual numbers) and reverse mode (tape) AD from scratch |
| `backprop_mlp.py` | 12.3 Automatic differentiation | Two-layer network on digits with hand-coded backpropagation, gradient check, SGD with momentum (`ch12_mlp_loss.pdf`) |
| `bias_variance_poly.py` | 12.4 Bias-variance | Monte Carlo bias^2 / variance / noise decomposition for polynomial regression (`ch12_bias_variance.pdf`) |
| `train_test_curves.py` | 12.4 Bias-variance | Training vs test error, tr(H) and ridge effective degrees of freedom (`ch12_train_test.pdf`) |
| `double_descent.py` | 12.4 Bias-variance | Double descent with random ReLU features and min-norm least squares (`ch12_double_descent.pdf`) |
| `ridge_svd.py` | 12.5 Regularisation | Ridge for p > n via normal equations, SVD and augmented data; bias and variance of ridge |
| `lasso_cd.py` | 12.5 Regularisation | Lasso by cyclic coordinate descent with soft-thresholding; KKT check; comparison with scikit-learn |
| `penalty_geometry.py` | 12.5 Regularisation | l1 vs l2 constrained least squares (`ch12_penalty_geometry.pdf`) |
| `enet_correlated.py` | 12.5 Regularisation | Lasso vs elastic net with groups of correlated predictors |
| `reg_paths.py` | 12.5 Regularisation | Ridge and lasso regularisation paths on standardised diabetes data (`ch12_reg_paths.pdf`) |
| `cv_lambda.py` | 12.6 Model selection | Hand-coded 10-fold CV for the lasso, one-standard-error rule (`ch12_cv_curve.pdf`) |
| `pipelines_ic.py` | 12.6 Model selection | Pipelines with StandardScaler + RidgeCV / LassoCV; lasso tuned by AIC and BIC |
| `forecast_factor.py` | 12.6 Model selection | Expanding-window pseudo-out-of-sample forecasts with a simulated factor panel: AR, OLS, ridge, lasso, PCR (`ch12_forecast_poos.pdf`) |
