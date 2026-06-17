import json
import os
import numpy as np
import pandas as pd
from scipy.stats import norm
import statsmodels.api as sm
from statsmodels.base.model import GenericLikelihoodModel

# All IO must use /app/data
DATA_PATH = "/app/data/REPExperiment1Data.dta"
OUTPUT_JSON = "/app/data/replication_results_experiment1_asian.json"
OUTPUT_CSV = "/app/data/replication_results_experiment1_asian.csv"
OUTPUT_LOG = "/app/data/replication_log.txt"


class IntervalRegression(GenericLikelihoodModel):
    """
    Interval regression with normal errors.
    Parameters estimated: beta (k_exog) and ln_sigma (1).
    Log-likelihood contributions handle interval-, left-, right-censoring and exact.
    """
    def __init__(self, endog, exog, lower, upper, **kwds):
        super().__init__(endog, exog, **kwds)
        self.lower = np.asarray(lower, dtype=float)
        self.upper = np.asarray(upper, dtype=float)
        # Masks for types
        self.is_left = np.isnan(self.lower) & np.isfinite(self.upper)
        self.is_right = np.isfinite(self.lower) & np.isnan(self.upper)
        self.is_interval = np.isfinite(self.lower) & np.isfinite(self.upper) & (self.upper > self.lower)
        self.is_exact = np.isfinite(self.lower) & np.isfinite(self.upper) & np.isclose(self.lower, self.upper)

    def nparams(self):
        return self.exog.shape[1] + 1

    def _unpack_params(self, params):
        beta = params[:-1]
        ln_sigma = params[-1]
        sigma = np.exp(ln_sigma)
        return beta, sigma, ln_sigma

    def loglikeobs(self, params):
        beta, sigma, ln_sigma = self._unpack_params(params)
        Xb = np.dot(self.exog, beta)
        ll = np.zeros(len(Xb), dtype=float)

        # Interval censored
        if self.is_interval.any():
            idx = self.is_interval
            a = (self.lower[idx] - Xb[idx]) / sigma
            b = (self.upper[idx] - Xb[idx]) / sigma
            L = np.clip(norm.cdf(b) - norm.cdf(a), 1e-12, None)
            ll[idx] = np.log(L)

        # Left-censored (upper bound only)
        if self.is_left.any():
            idx = self.is_left
            b = (self.upper[idx] - Xb[idx]) / sigma
            L = np.clip(norm.cdf(b), 1e-12, None)
            ll[idx] = np.log(L)

        # Right-censored (lower bound only)
        if self.is_right.any():
            idx = self.is_right
            a = (self.lower[idx] - Xb[idx]) / sigma
            L = np.clip(1.0 - norm.cdf(a), 1e-12, None)
            ll[idx] = np.log(L)

        # Exact
        if self.is_exact.any():
            idx = self.is_exact
            y = self.lower[idx]
            z = (y - Xb[idx]) / sigma
            ll[idx] = norm.logpdf(z) - ln_sigma

        return ll

    def score_obs(self, params):
        beta, sigma, ln_sigma = self._unpack_params(params)
        Xb = np.dot(self.exog, beta)
        k = self.exog.shape[1]
        sco = np.zeros((len(Xb), k + 1), dtype=float)

        # Helpers per type
        # Interval censored
        if self.is_interval.any():
            idx = self.is_interval
            Xi = self.exog[idx]
            a = (self.lower[idx] - Xb[idx]) / sigma
            b = (self.upper[idx] - Xb[idx]) / sigma
            Phi_a = norm.cdf(a)
            Phi_b = norm.cdf(b)
            phi_a = norm.pdf(a)
            phi_b = norm.pdf(b)
            L = np.clip(Phi_b - Phi_a, 1e-12, None)
            dlogL_dmu = (phi_a - phi_b) / (sigma * L)
            dlogL_dlns = (-b * phi_b + a * phi_a) / L
            sco[idx, :k] = (dlogL_dmu[:, None]) * Xi
            sco[idx, k] = dlogL_dlns

        # Left-censored
        if self.is_left.any():
            idx = self.is_left
            Xi = self.exog[idx]
            b = (self.upper[idx] - Xb[idx]) / sigma
            Phi_b = np.clip(norm.cdf(b), 1e-12, None)
            phi_b = norm.pdf(b)
            dlogL_dmu = -(phi_b / (sigma * Phi_b))
            dlogL_dlns = -(b * phi_b) / Phi_b
            sco[idx, :k] = (dlogL_dmu[:, None]) * Xi
            sco[idx, k] = dlogL_dlns

        # Right-censored
        if self.is_right.any():
            idx = self.is_right
            Xi = self.exog[idx]
            a = (self.lower[idx] - Xb[idx]) / sigma
            one_minus_Phi_a = np.clip(1.0 - norm.cdf(a), 1e-12, None)
            phi_a = norm.pdf(a)
            dlogL_dmu = -(phi_a / (sigma * one_minus_Phi_a))
            dlogL_dlns = (a * phi_a) / one_minus_Phi_a
            sco[idx, :k] = (dlogL_dmu[:, None]) * Xi
            sco[idx, k] = dlogL_dlns

        # Exact
        if self.is_exact.any():
            idx = self.is_exact
            Xi = self.exog[idx]
            y = self.lower[idx]
            z = (y - Xb[idx]) / sigma
            dlogL_dmu = (y - Xb[idx]) / (sigma ** 2)
            dlogL_dlns = -1.0 + (z ** 2)
            sco[idx, :k] = (dlogL_dmu[:, None]) * Xi
            sco[idx, k] = dlogL_dlns

        return sco


def main():
    os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
    log_lines = []
    try:
        df = pd.read_stata(DATA_PATH)
        log_lines.append(f"Loaded data: {DATA_PATH} with shape {df.shape}")
    except Exception as e:
        with open(OUTPUT_LOG, "w") as f:
            f.write("Failed to load data: " + str(e))
        raise

    # Filter to Asian subjects
    df_asian = df[df["asian"] == 1].copy()
    # Drop rows with both bounds missing
    df_asian = df_asian[~(df_asian["lndiscratelo"].isna() & df_asian["lndiscratehi"].isna())]

    # Construct design matrix as in Stata intreg call
    # intreg lndiscratelo lndiscratehi givenprimingquestionnaire largestakes longterm largelong if asian == 1, cluster(id)
    X = df_asian[["givenprimingquestionnaire", "largestakes", "longterm", "largelong"]].copy()
    X = sm.add_constant(X, has_constant='add')

    lower = df_asian["lndiscratelo"].to_numpy()
    upper = df_asian["lndiscratehi"].to_numpy()

    # For GenericLikelihoodModel we need an endog placeholder
    endog = np.zeros(len(df_asian))

    # Initialize via OLS on midpoint/imputed
    y_init = np.where(np.isnan(lower), upper,
               np.where(np.isnan(upper), lower, (lower + upper) / 2.0))

    # Fit custom interval regression
    model = IntervalRegression(endog, X.values, lower, upper)

    # Start params: OLS on y_init
    ols_res = sm.OLS(y_init, X.values, missing='drop').fit()
    beta0 = ols_res.params
    resid = y_init - ols_res.predict(X.values)
    sigma0 = np.nanstd(resid)
    if not np.isfinite(sigma0) or sigma0 <= 0:
        sigma0 = 1.0
    start_params = np.r_[beta0, np.log(sigma0)]

    res = model.fit(start_params=start_params, method='bfgs', disp=False, maxiter=500)

    # Cluster-robust by id (sandwich with cluster)
    from statsmodels.stats.sandwich_covariance import cov_cluster
    groups = df_asian["id"].to_numpy()
    cov = cov_cluster(res, groups)
    params = res.params
    bse_arr = np.sqrt(np.diag(cov))
    z_vals = params / bse_arr
    pvals_arr = 2 * (1 - norm.cdf(np.abs(z_vals)))

    # Extract results for treatment coefficient
    param_names = list(X.columns) + ["ln_sigma"]
    coef = dict(zip(param_names, params))
    se = dict(zip(param_names, bse_arr))
    pvalues = dict(zip(param_names, pvals_arr))

    treat_name = "givenprimingquestionnaire"
    out = {
        "model": "Interval regression (normal errors) with cluster-robust SE (clusters: id)",
        "sample": "Asian subjects only (asian==1)",
        "n_obs": int(df_asian.shape[0]),
        "n_clusters": int(pd.Series(groups).nunique()),
        "variables": param_names,
        "coef": coef,
        "se": se,
        "pvalues": pvalues,
        "treatment_coef": coef.get(treat_name, None),
        "treatment_se": se.get(treat_name, None),
        "treatment_pvalue": pvalues.get(treat_name, None)
    }

    # Save outputs
    with open(OUTPUT_JSON, "w") as f:
        json.dump(out, f, indent=2)

    # Also save a CSV summary for easy reading
    df_out = pd.DataFrame({
        "variable": param_names,
        "coef": [coef.get(n) for n in param_names],
        "se": [se.get(n) for n in param_names],
        "pvalue": [pvalues.get(n) for n in param_names]
    })
    df_out.to_csv(OUTPUT_CSV, index=False)

    # Log
    log_lines.append("Fitted interval regression with cluster-robust SE (id)")
    log_lines.append(f"Treatment ({treat_name}) coef: {out['treatment_coef']}")
    log_lines.append(f"Treatment SE: {out['treatment_se']}")
    log_lines.append(f"Treatment p-value: {out['treatment_pvalue']}")
    with open(OUTPUT_LOG, "w") as f:
        f.write("\n".join(log_lines))


if __name__ == "__main__":
    main()
