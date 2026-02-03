import os
# Limit math library threads to improve stability under emulation
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
import json
import numpy as np
import pandas as pd
from pathlib import Path

OUTPUT_DIR = "/app/data"
# Detect data directory: prefer local script folder (mounted at /workspace/replication_data),
# fall back to /app/data/original/7/data-only/replication_data
SCRIPT_DIR = Path(__file__).resolve().parent
CANDIDATE_DIRS = [SCRIPT_DIR, Path("/app/data/original/7/data-only/replication_data")]
for p in CANDIDATE_DIRS:
    if (p / "compiled.dta").exists():
        DATA_DIR = str(p)
        break
else:
    # Default to script dir; will raise if file truly missing
    DATA_DIR = str(SCRIPT_DIR)


class SimpleResult:
    def __init__(self, name, params, std_errors, pvalues, nobs, df_resid, param_names):
        self.name = name
        self.params = pd.Series(params, index=param_names)
        self.std_errors = pd.Series(std_errors, index=param_names)
        self.pvalues = pd.Series(pvalues, index=param_names)
        self.nobs = nobs
        self.df_resid = df_resid

    @property
    def summary(self):
        lines = [f"Model: {self.name}"]
        lines.append(f"N={self.nobs}, df_resid={self.df_resid}")
        lines.append("Variable\tCoef\tStd.Err\tp-value")
        for var in self.params.index:
            lines.append(f"{var}\t{self.params[var]:.6f}\t{self.std_errors[var]:.6f}\t{self.pvalues[var]:.6f}")
        return "\n".join(lines)


def norm_cdf(x):
    # Standard normal CDF using error function
    return 0.5 * (1.0 + np.math.erf(x / np.sqrt(2.0)))


def prepare_data():
    # Load compiled dataset
    compiled_path = os.path.join(DATA_DIR, "compiled.dta")
    df = pd.read_stata(compiled_path)

    # Harmonize identifiers: Year appears as 7..16 representing 2007..2016
    df["year_actual"] = 2000 + df["year"].astype(int)

    # Filter to match the original study period 2007-2013
    df = df[(df["year_actual"] >= 2007) & (df["year_actual"] <= 2013)].copy()

    # Ensure State is string identifier
    if "State" not in df.columns:
        raise ValueError("Expected column 'State' not found in compiled.dta")

    # Create log-transformed variables for continuous measures
    log_vars = {
        "carbon_adj": "ln_carbon_adj",
        "wrkhrs": "ln_wrkhrs",
        "laborprod": "ln_laborprod",
        "emppop": "ln_emppop",
        "pop": "ln_pop",
        "energy": "ln_energy",
        "manuf": "ln_manuf",
        "gdp": "ln_gdp",
        "rgdp": "ln_rgdp",
    }

    # Drop non-positive values for variables to be logged
    for raw, logn in log_vars.items():
        if raw in df.columns:
            df = df[df[raw] > 0]

    # Log transform
    for raw, logn in log_vars.items():
        if raw in df.columns:
            df[logn] = np.log(df[raw])

    # Set panel index
    df = df.set_index(["State", "year_actual"]).sort_index()

    return df


def build_fe_design(dfr, y_col, x_cols):
    # Build design matrix with intercept + state FE + year FE (drop first of each to avoid collinearity)
    # Construct category lists
    states = sorted(dfr["State"].astype(str).unique().tolist())
    years = sorted(dfr["year_actual"].astype(int).unique().tolist())
    base_state = states[0]
    base_year = years[0]
    state_dummies = [s for s in states if s != base_state]
    year_dummies = [y for y in years if y != base_year]

    names = []
    # Main regressors
    for col in x_cols:
        names.append(col)
    # Intercept
    names.append("const")
    # State FE names
    names.extend([f"state_{s}" for s in state_dummies])
    # Year FE names
    names.extend([f"year_{y}" for y in year_dummies])

    # Build rows
    X = []
    y = []
    groups = []
    for _, row in dfr.iterrows():
        xi = []
        for col in x_cols:
            xi.append(float(row[col]))
        xi.append(1.0)  # intercept
        # state FE
        s_val = str(row["State"]) 
        for s in state_dummies:
            xi.append(1.0 if s_val == s else 0.0)
        # year FE
        y_val = int(row["year_actual"]) 
        for ycat in year_dummies:
            xi.append(1.0 if y_val == ycat else 0.0)
        X.append(xi)
        y.append(float(row[y_col]))
        groups.append(s_val)

    return y, X, names, groups


def ols_cluster_state(y, X, groups):
    # Solve OLS via normal equations using pure Python
    N = len(y)
    K = len(X[0]) if N > 0 else 0

    # Compute XtX and Xty
    XtX = [[0.0 for _ in range(K)] for _ in range(K)]
    Xty = [0.0 for _ in range(K)]
    for i in range(N):
        xi = X[i]
        yi = y[i]
        for a in range(K):
            Xty[a] += xi[a] * yi
            va = xi[a]
            for b in range(K):
                XtX[a][b] += va * xi[b]

    # Gaussian elimination to solve XtX * beta = Xty
    def gauss_solve(A, b):
        n = len(b)
        # Augment matrix
        M = [A[i][:] + [b[i]] for i in range(n)]
        # Forward elimination with partial pivoting
        for k in range(n):
            # pivot
            piv = max(range(k, n), key=lambda r: abs(M[r][k]))
            if abs(M[piv][k]) < 1e-12:
                continue
            if piv != k:
                M[k], M[piv] = M[piv], M[k]
            # eliminate
            pivot = M[k][k]
            for j in range(k, n + 1):
                M[k][j] /= pivot
            for i in range(k + 1, n):
                factor = M[i][k]
                if factor != 0.0:
                    for j in range(k, n + 1):
                        M[i][j] -= factor * M[k][j]
        # Back substitution
        x = [0.0] * n
        for i in range(n - 1, -1, -1):
            x[i] = M[i][n] - sum(M[i][j] * x[j] for j in range(i + 1, n))
        return x

    # Gauss-Jordan to invert XtX
    def gauss_jordan_inverse(A):
        n = len(A)
        # Create augmented matrix [A | I]
        M = [A[i][:] + [0.0] * n for i in range(n)]
        for i in range(n):
            M[i][n + i] = 1.0
        # Forward elimination
        for k in range(n):
            # pivot
            piv = max(range(k, n), key=lambda r: abs(M[r][k]))
            if abs(M[piv][k]) < 1e-12:
                continue
            if piv != k:
                M[k], M[piv] = M[piv], M[k]
            pivot = M[k][k]
            # normalize row
            for j in range(2 * n - 1, k - 1, -1):
                M[k][j] /= pivot
            # eliminate other rows
            for i in range(n):
                if i == k:
                    continue
                factor = M[i][k]
                if factor != 0.0:
                    for j in range(2 * n - 1, k - 1, -1):
                        M[i][j] -= factor * M[k][j]
        # Extract inverse
        inv = [row[n:] for row in M]
        return inv

    beta = gauss_solve(XtX, Xty)

    # Residuals e = y - X beta
    e = [0.0] * N
    for i in range(N):
        xb = 0.0
        xi = X[i]
        for a in range(K):
            xb += xi[a] * beta[a]
        e[i] = y[i] - xb

    # Clusters by state
    clusters = {}
    for i, g in enumerate(groups):
        clusters.setdefault(g, []).append(i)
    G = len(clusters)

    # Meat matrix
    meat = [[0.0 for _ in range(K)] for _ in range(K)]
    for idx in clusters.values():
        Sg = [0.0] * K
        for i in idx:
            xi = X[i]
            ei = e[i]
            for a in range(K):
                Sg[a] += xi[a] * ei
        # outer product Sg Sg'
        for a in range(K):
            va = Sg[a]
            for b in range(K):
                meat[a][b] += va * Sg[b]

    # Bread inverse
    XtX_inv = gauss_jordan_inverse(XtX)

    # Variance-covariance: V = c * XtX_inv @ meat @ XtX_inv
    # matrix multiply helper
    def matmul(A, B):
        n = len(A); m = len(A[0]); p = len(B[0])
        C = [[0.0 for _ in range(p)] for _ in range(n)]
        for i in range(n):
            Ai = A[i]
            for k in range(m):
                aik = Ai[k]
                if aik == 0.0:
                    continue
                Bk = B[k]
                for j in range(p):
                    C[i][j] += aik * Bk[j]
        return C

    Nobs = N
    df_resid = N - K
    if G > 1 and Nobs > K:
        c = (G / (G - 1.0)) * ((Nobs - 1.0) / (Nobs - K))
    else:
        c = 1.0

    A = matmul(XtX_inv, meat)
    V = matmul(A, XtX_inv)
    # scale
    for i in range(K):
        for j in range(K):
            V[i][j] *= c

    # standard errors
    se = [0.0] * K
    for i in range(K):
        se[i] = (V[i][i] if V[i][i] > 0 else 0.0) ** 0.5

    # p-values using normal approximation    # p-values using normal approximation
    pvals = []
    for i in range(K):
        if se[i] == 0:
            pvals.append(1.0)
            continue
        tval = abs(beta[i] / se[i])
        cdf = norm_cdf(tval)
        pvals.append(2 * (1 - cdf))

    return beta, se, pvals, df_resid    return beta, se, pvals, df_residdef ols_cluster_state(y, X, groups):
    # OLS coefficients
    # Use lstsq for stability
    beta, residuals_sum, rank, s = np.linalg.lstsq(X, y, rcond=None)
    beta = beta.reshape(-1)
    y_hat = X @ beta.reshape(-1, 1)
    e = (y - y_hat).reshape(-1)

    # Bread
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.inv(XtX)
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)

    # Meat (cluster by state)
    clusters = {}
    for i, g in enumerate(groups):
        clusters.setdefault(g, []).append(i)
    G = len(clusters)
    N, k = X.shape

    meat = np.zeros((k, k))
    for idx in clusters.values():
        Xg = X[idx, :]
        eg = e[idx].reshape(-1, 1)
        Sg = Xg.T @ eg  # k x 1
        meat += Sg @ Sg.T  # k x k

    # Small sample correction (CR1)
    if G > 1 and N > k:
        df_correction = (G / (G - 1)) * ((N - 1) / (N - k))
    else:
        df_correction = 1.0

    V = df_correction * XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))

    # t-stats and normal-approx p-values
    tvals = beta / se
    pvals = 2 * (1 - np.vectorize(norm_cdf)(np.abs(tvals)))

    df_resid = N - k
    return beta, se, pvals, df_resid


def run_models(df):
    results = {}
    dfr = df.reset_index()

    # Model 1: Scale effect
    x1 = ["ln_wrkhrs", "ln_laborprod", "ln_emppop", "ln_pop", "ln_energy", "ln_manuf"]
    y, X, names, groups = build_fe_design(dfr, "ln_carbon_adj", x1)
    beta, se, pvals, df_resid = ols_cluster_state(y, X, groups)
    res1 = SimpleResult("model_scale_fe", beta, se, pvals, nobs=int(dfr.shape[0]), df_resid=int(df_resid), param_names=names)
    results[res1.name] = res1

    # Model 2: Composition effect
    gdp_term = "ln_rgdp" if "ln_rgdp" in dfr.columns else ("ln_gdp" if "ln_gdp" in dfr.columns else None)
    if gdp_term is not None:
        x2 = ["ln_wrkhrs", gdp_term, "ln_pop", "ln_energy", "ln_manuf"]
    else:
        x2 = ["ln_wrkhrs", "ln_pop", "ln_energy", "ln_manuf"]
    y2, X2, names2, groups2 = build_fe_design(dfr, "ln_carbon_adj", x2)
    beta2, se2, pvals2, df_resid2 = ols_cluster_state(y2, X2, groups2)
    res2 = SimpleResult("model_composition_fe", beta2, se2, pvals2, nobs=int(dfr.shape[0]), df_resid=int(df_resid2), param_names=names2)
    results[res2.name] = res2

    return results


def compute_diff_correlation(df):
    # Compute within-state first differences of log variables
    work = df[["ln_wrkhrs", "ln_carbon_adj"]].copy()
    work = work.groupby(level=0).apply(lambda g: g.sort_index(level=1).diff()).reset_index(level=0, drop=True)
    work = work.dropna()
    if work.empty:
        return None
    corr = work["ln_wrkhrs"].corr(work["ln_carbon_adj"])
    return float(corr)


def save_outputs(results, corr, df):
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    # Save model summaries
    summary_path = os.path.join(OUTPUT_DIR, "replication_results_summary.txt")
    with open(summary_path, "w") as f:
        for name, res in results.items():
            f.write(f"=== {name} ===\n")
            f.write(str(res.summary))
            f.write("\n\n")
        f.write("=== First-Difference Correlation ===\n")
        f.write(f"corr(delta ln_wrkhrs, delta ln_carbon_adj) = {corr}\n")

    # Save coefficients table
    rows = []
    for name, res in results.items():
        for var in res.params.index:
            rows.append({
                "model": name,
                "variable": var,
                "coef": float(res.params[var]),
                "std_err": float(res.std_errors[var]),
                "p_value": float(res.pvalues[var]),
            })
    coef_path = os.path.join(OUTPUT_DIR, "replication_coefficients.csv")
    pd.DataFrame(rows).to_csv(coef_path, index=False)

    # Save sample info
    info = {
        "n_observations": int(df.shape[0]),
        "n_states": int(df.index.get_level_values(0).nunique()),
        "years": sorted(list(set(df.index.get_level_values(1).tolist()))),
        "notes": "Two-way FE (state & year) via OLS with dummies. One-way cluster-robust SEs (state). Variables log-transformed; period 2007-2013."
    }
    info_path = os.path.join(OUTPUT_DIR, "replication_sample_info.json")
    with open(info_path, "w") as f:
        json.dump(info, f)


if __name__ == "__main__":
    df = prepare_data()
    results = run_models(df)
    corr = compute_diff_correlation(df)
    save_outputs(results, corr, df)
    print("Replication analysis complete. Outputs saved to:")
    print(" - /app/data/replication_results_summary.txt")
    print(" - /app/data/replication_coefficients.csv")
    print(" - /app/data/replication_sample_info.json")
