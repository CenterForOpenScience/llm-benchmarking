import json
import os
import pandas as pd
import numpy as np
from scipy.stats import pearsonr, t
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


def residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    model = LinearRegression()
    model.fit(X, y)
    return (y - model.predict(X)).ravel()


def partial_corr(x, y, C):
    x_res = residualize(x, C)
    y_res = residualize(y, C)
    r, p = pearsonr(x_res, y_res)
    return float(r), float(p), int(len(x_res))


def compute_partial_corr_and_pvalues(df_net: pd.DataFrame):
    # Z-score columns for stability
    Z = (df_net - df_net.mean()) / df_net.std(ddof=0)
    Z = Z.dropna()
    n, d = Z.shape
    # Empirical correlation matrix
    R = np.corrcoef(Z.values, rowvar=False)
    # Pseudo-inverse precision to handle potential singularity
    P = np.linalg.pinv(R)
    # Partial correlations from precision
    pcorr = np.zeros((d, d))
    for i in range(d):
        for j in range(d):
            if i == j:
                pcorr[i, j] = 1.0
            else:
                pcorr[i, j] = -P[i, j] / np.sqrt(P[i, i] * P[j, j])
    # p-values for partial correlations controlling for remaining (d-2) variables
    # df = n - d (standard result for partial correlation testing)
    dfree = max(n - d, 1)
    pvals = np.ones((d, d))
    for i in range(d):
        for j in range(i + 1, d):
            r = np.clip(pcorr[i, j], -0.999999, 0.999999)
            t_stat = r * np.sqrt(dfree / (1 - r**2))
            pval = 2 * t.sf(np.abs(t_stat), dfree)
            pvals[i, j] = pval
            pvals[j, i] = pval
    return Z.columns.tolist(), pcorr, pvals, n, dfree


def main():
    data_path = "/app/data/Dataset.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Required dataset not found at {data_path}")

    df = pd.read_csv(data_path)

    # Task1.1: Partial correlations controlling for Age
    required = ["MiniK_Total", "HKSS_Total", "DSM5_Total", "Age"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Dataset missing required columns: {miss}")

    dsub = df[required].dropna()
    X = dsub[["Age"]].to_numpy()

    r_mini_dsm, p_mini_dsm, n = partial_corr(dsub["MiniK_Total"].to_numpy().astype(float),
                                             dsub["DSM5_Total"].to_numpy().astype(float), X)
    r_hkss_dsm, p_hkss_dsm, _ = partial_corr(dsub["HKSS_Total"].to_numpy().astype(float),
                                             dsub["DSM5_Total"].to_numpy().astype(float), X)
    r_mini_hkss, p_mini_hkss, _ = partial_corr(dsub["MiniK_Total"].to_numpy().astype(float),
                                               dsub["HKSS_Total"].to_numpy().astype(float), X)

    # Task1.2: PCA of MiniK_Total and HKSS_Total, correlate with DSM5_Total controlling for Age
    pca_input_df = dsub[["MiniK_Total", "HKSS_Total"]].dropna()
    pca_input = pca_input_df.to_numpy().astype(float)
    pca = PCA(n_components=1)
    Z2 = (pca_input - pca_input.mean(axis=0)) / pca_input.std(axis=0, ddof=0)
    comp = pca.fit_transform(Z2).ravel()

    # Orient PC1 to correlate positively with MiniK_Total for interpretability
    corr_pc1_minik = np.corrcoef(comp, pca_input_df["MiniK_Total"].to_numpy().astype(float))[0, 1]
    if np.isnan(corr_pc1_minik):
        corr_pc1_minik = 0.0
    if corr_pc1_minik < 0:
        comp = -comp

    # Align DSM5 and Age to PCA rows
    idx = pca_input_df.index
    dsm5_pca = dsub.loc[idx, "DSM5_Total"].to_numpy().astype(float)
    age_pca = dsub.loc[idx, "Age"].to_numpy().astype(float).reshape(-1, 1)
    r_pca_dsm, p_pca_dsm, n_pca = partial_corr(comp, dsm5_pca, age_pca)

    # Task1.3: Partial correlation network among specified variables
    net_vars = [
        "MiniK_Total", "HKSS_Total", "DSM5_Total", "Age",
        "Bio_Sib", "Half_Sib", "Step_Sib", "Stepparent",
        "SH_Total", "Attach_Total", "Aggresion_Total"
    ]
    net_present = [v for v in net_vars if v in df.columns]
    df_net = df[net_present].dropna()
    net_cols, pcorr_mat, pval_mat, n_net, dfree = compute_partial_corr_and_pvalues(df_net)

    # Bonferroni threshold for 11 variables: 55 pairs
    total_tests = 55
    alpha_bonf = 0.05 / total_tests

    significant_edges = []
    all_edges = []
    d = len(net_cols)
    for i in range(d):
        for j in range(i + 1, d):
            edge = {
                "var_i": net_cols[i],
                "var_j": net_cols[j],
                "partial_r": float(pcorr_mat[i, j]),
                "p_value": float(pval_mat[i, j])
            }
            all_edges.append(edge)
            if pval_mat[i, j] < alpha_bonf:
                significant_edges.append({**edge, "bonferroni_significant": True})

    results = {
        "task_id": "Task1",
        "analyses": {
            "age_controlled_partial_correlations": {
                "MiniK_DSM5": {"r": r_mini_dsm, "p": p_mini_dsm, "n": n},
                "HKSS_DSM5": {"r": r_hkss_dsm, "p": p_hkss_dsm, "n": n},
                "MiniK_HKSS": {"r": r_mini_hkss, "p": p_mini_hkss, "n": n}
            },
            "pca_component_partial_correlation": {
                "PC1_orientation": "oriented_to_correlate_positively_with_MiniK_Total",
                "PC1_life_history_vs_DSM5": {"r": r_pca_dsm, "p": p_pca_dsm, "n": n_pca},
                "explained_variance_ratio_PC1": float(pca.explained_variance_ratio_[0])
            },
            "partial_correlation_network": {
                "variables": net_cols,
                "n": n_net,
                "df": dfree,
                "total_tests": total_tests,
                "bonferroni_alpha": alpha_bonf,
                "partial_correlation_matrix": pcorr_mat.tolist(),
                "significant_edges": significant_edges
            }
        }
    }

    out_path = "/app/data/Multi100-CCBE4-Task1_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(json.dumps({
        "outputs": out_path,
        "summary": {
            "MiniK_DSM5": results["analyses"]["age_controlled_partial_correlations"]["MiniK_DSM5"],
            "HKSS_DSM5": results["analyses"]["age_controlled_partial_correlations"]["HKSS_DSM5"],
            "PC1_vs_DSM5": results["analyses"]["pca_component_partial_correlation"]["PC1_life_history_vs_DSM5"],
            "n_significant_network_edges": len(significant_edges)
        }
    }))


if __name__ == "__main__":
    main()
