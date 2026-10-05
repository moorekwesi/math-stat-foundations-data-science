# DEP 803 -- Chapter 12: Matrix Calculus, Regularisation and Bias-Variance
# sklearn Pipelines (StandardScaler + RidgeCV / LassoCV) and AIC/BIC selection.
# (c) 2026 Stephen E. Moore. MIT Licence.
import warnings
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.linear_model import LassoCV, LassoLarsIC, LinearRegression, RidgeCV
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler

warnings.filterwarnings("ignore")        # LARS early-stopping notices
X, y = load_diabetes(return_X_y=True)
# add squares and interactions: 10 -> 65 predictors, a "many predictors" setting
X = PolynomialFeatures(degree=2, include_bias=False).fit_transform(X)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=803)
print(f"training n = {len(ytr)}, predictors p = {X.shape[1]}")

models = {
    "OLS": make_pipeline(StandardScaler(), LinearRegression()),
    "RidgeCV": make_pipeline(StandardScaler(),
                             RidgeCV(alphas=np.logspace(-2, 4, 60))),
    "LassoCV": make_pipeline(StandardScaler(),
                             LassoCV(cv=10, random_state=803, max_iter=50000)),
    "Lasso-AIC": make_pipeline(StandardScaler(), LassoLarsIC(criterion="aic",
                                                             normalize=False)),
    "Lasso-BIC": make_pipeline(StandardScaler(), LassoLarsIC(criterion="bic",
                                                             normalize=False)),
}
print(f"{'model':<10}{'penalty':>9}{'nonzero':>9}{'CV RMSE':>9}{'test RMSE':>11}")
for name, pipe in models.items():
    cv = -cross_val_score(pipe, Xtr, ytr, cv=5,
                          scoring="neg_root_mean_squared_error").mean()
    pipe.fit(Xtr, ytr)
    est = pipe[-1]
    pen = getattr(est, "alpha_", np.nan)
    nz = np.sum(np.abs(est.coef_) > 1e-10)
    rmse = np.sqrt(np.mean((yte - pipe.predict(Xte)) ** 2))
    print(f"{name:<10}{pen:>9.3f}{nz:>9d}{cv:>9.2f}{rmse:>11.2f}")
