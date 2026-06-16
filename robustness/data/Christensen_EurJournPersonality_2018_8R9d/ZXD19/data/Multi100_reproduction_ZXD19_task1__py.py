import os
import json
import math
import random
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats
import statsmodels.api as sm

# Reproducibility
random.seed(42)
np.random.seed(42)

DATA_DIR = "/app/data"
ARTIFACTS_DIR = "/app/artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Utility functions

def normalize_token(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    # Remove non-letters
    s = "".join([ch for ch in s if ch.isalpha()])
    # Drop common placeholders
    if s in ("", "na", "nana", "nan", "none", "dontknow", "dk", "n", "idk"):
        return ""
    return s


def load_data():
    flu_path = os.path.join(DATA_DIR, "FINAL fluency.csv")
    open_path = os.path.join(DATA_DIR, "FINAL open.csv")
    flu = pd.read_csv(flu_path)
    opn = pd.read_csv(open_path)
    return flu, opn


def build_binary_matrix(flu: pd.DataFrame) -> pd.DataFrame:
    # Collect tokens per participant as a set
    resp_cols = [c for c in flu.columns if c.startswith("vf_an_")]
    token_sets = []
    vocab = set()
    for _, row in flu.iterrows():
        toks = [normalize_token(row[c]) for c in resp_cols]
        toks = [t for t in toks if t != ""]
        s = set(toks)
        token_sets.append(s)
        vocab.update(s)
    vocab = sorted(vocab)
    idx = {w: i for i, w in enumerate(vocab)}
    mat = np.zeros((len(token_sets), len(vocab)), dtype=np.uint8)
    for r, s in enumerate(token_sets):
        for w in s:
            mat[r, idx[w]] = 1
    bin_df = pd.DataFrame(mat, columns=vocab)
    return bin_df


def filter_common_words(bin1: pd.DataFrame, bin2: pd.DataFrame, min_count: int = 2):
    c1 = (bin1.sum(axis=0) >= min_count)
    c2 = (bin2.sum(axis=0) >= min_count)
    common = bin1.columns[c1 & c2]
    return bin1[common], bin2[common]


def cosine_similarity_from_binary(X: np.ndarray) -> np.ndarray:
    # X: n_participants x n_words binary
    # cos(i,j) = (Xi dot Xj) / (||Xi|| * ||Xj||) where vectors are columns of X
    # Compute column-wise norms
    Xt = X.T  # n_words x n_participants
    dots = Xt @ X  # n_words x n_words
    norms = np.sqrt(np.diag(dots))
    denom = norms[:, None] * norms[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        cos = np.true_divide(dots, denom)
        cos[~np.isfinite(cos)] = 0.0
    np.fill_diagonal(cos, 0.0)  # no self loops
    return cos


def greedy_planar_backbone(weights: np.ndarray) -> np.ndarray:
    # Build a greedy planar backbone by adding edges in descending order while keeping planarity
    n = weights.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    # collect candidate edges with weight > 0
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            w = float(weights[i, j])
            if w > 0:
                edges.append((i, j, w))
    edges.sort(key=lambda x: x[2], reverse=True)
    max_edges = max(0, 3*n - 6)  # planar graph edge upper bound
    for (u, v, w) in edges:
        if G.number_of_edges() >= max_edges:
            break
        G.add_edge(u, v, weight=w)
        planar, _ = nx.check_planarity(G)
        if not planar:
            G.remove_edge(u, v)
    # Convert to adjacency matrix (unweighted)
    A = np.zeros((n, n), dtype=np.uint8)
    for u, v in G.edges():
        A[u, v] = 1
        A[v, u] = 1
    return A


def largest_cc_aspl_from_adj(A: np.ndarray) -> float:
    # Build graph and compute ASPL of largest connected component
    n = A.shape[0]
    if n == 0:
        return float('nan')
    G = nx.from_numpy_array(A)
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return float('nan')
    components = list(nx.connected_components(G))
    if len(components) == 0:
        return float('nan')
    lcc_nodes = max(components, key=len)
    if len(lcc_nodes) < 2:
        return float('nan')
    lcc = G.subgraph(lcc_nodes).copy()
    try:
        return nx.average_shortest_path_length(lcc)
    except Exception:
        return float('nan')


def approach1_group_bootstrap(bin_df: pd.DataFrame, groups: pd.Series, node_rate: float = 0.9, n_boot: int = 1000):
    # Split by group and build common-word matrices
    bin1 = bin_df.loc[groups.values]
    bin2 = bin_df.loc[~groups.values]
    if bin1.shape[0] == 0 or bin2.shape[0] == 0:
        raise RuntimeError("One of the groups is empty after split.")
    bin1c, bin2c = filter_common_words(bin1, bin2, min_count=2)
    # If not enough common words, return NaNs
    if bin1c.shape[1] < 3:
        return {
            "mean_aspl_high": float('nan'),
            "mean_aspl_low": float('nan'),
            "t_stat": float('nan'),
            "p_value": float('nan'),
            "n_nodes": int(bin1c.shape[1])
        }
    # Cosine similarity and planar backbone once per group
    cos1 = cosine_similarity_from_binary(bin1c.values)
    cos2 = cosine_similarity_from_binary(bin2c.values)
    A1 = greedy_planar_backbone(cos1)
    A2 = greedy_planar_backbone(cos2)
    n_nodes = A1.shape[0]
    k = max(2, int(round(node_rate * n_nodes)))
    aspl1 = []
    aspl2 = []
    for i in range(n_boot):
        nodes = np.random.choice(n_nodes, size=k, replace=False)
        A1s = A1[np.ix_(nodes, nodes)]
        A2s = A2[np.ix_(nodes, nodes)]
        a1 = largest_cc_aspl_from_adj(A1s)
        a2 = largest_cc_aspl_from_adj(A2s)
        if np.isfinite(a1) and np.isfinite(a2):
            aspl1.append(a1)
            aspl2.append(a2)
    if len(aspl1) == 0:
        return {
            "mean_aspl_high": float('nan'),
            "mean_aspl_low": float('nan'),
            "t_stat": float('nan'),
            "p_value": float('nan'),
            "n_nodes": int(n_nodes)
        }
    aspl1 = np.array(aspl1)
    aspl2 = np.array(aspl2)
    t_stat, p_val = stats.ttest_rel(aspl1, aspl2, nan_policy='omit')
    return {
        "mean_aspl_high": float(np.nanmean(aspl1)),
        "mean_aspl_low": float(np.nanmean(aspl2)),
        "t_stat": float(t_stat) if np.isfinite(t_stat) else float('nan'),
        "p_value": float(p_val) if np.isfinite(p_val) else float('nan'),
        "n_nodes": int(n_nodes)
    }


def approach2_participant_bootstrap(bin_df: pd.DataFrame, latent: pd.Series, n_boot: int = 1000, sub_n: int = 20):
    rows = bin_df.shape[0]
    results = []
    for b in range(n_boot):
        ids = np.random.choice(rows, size=min(sub_n, rows), replace=False)
        subX = bin_df.iloc[ids, :]
        mean_open = float(latent.iloc[ids].mean())
        # Filter words that appear in at least 2 participants
        common_mask = (subX.sum(axis=0) >= 2)
        common_cols = subX.columns[common_mask]
        if len(common_cols) < 3:
            continue
        X = subX[common_cols].values
        cos = cosine_similarity_from_binary(X)
        A = greedy_planar_backbone(cos)
        aspl = largest_cc_aspl_from_adj(A)
        if np.isfinite(aspl):
            results.append({
                "open": mean_open,
                "aspl": float(aspl),
                "num_words": int(A.shape[0])
            })
    if len(results) == 0:
        return {
            "n_samples": 0,
            "pearson_r": float('nan'),
            "pearson_p": float('nan'),
            "ols_open_coef": float('nan'),
            "ols_open_p": float('nan'),
            "ols2_open_coef": float('nan'),
            "ols2_open_p": float('nan')
        }
    df = pd.DataFrame(results)
    r, p = stats.pearsonr(df["open"], df["aspl"]) if len(df) > 1 else (np.nan, np.nan)
    # OLS aspl ~ open
    try:
        X1 = sm.add_constant(df[["open"]])
        ols1 = sm.OLS(df["aspl"], X1, missing='drop').fit()
        ols_open_coef = float(ols1.params.get("open", np.nan))
        ols_open_p = float(ols1.pvalues.get("open", np.nan))
    except Exception:
        ols_open_coef = float('nan')
        ols_open_p = float('nan')
    # OLS aspl ~ open + num_words
    try:
        X2 = sm.add_constant(df[["open", "num_words"]])
        ols2 = sm.OLS(df["aspl"], X2, missing='drop').fit()
        ols2_open_coef = float(ols2.params.get("open", np.nan))
        ols2_open_p = float(ols2.pvalues.get("open", np.nan))
    except Exception:
        ols2_open_coef = float('nan')
        ols2_open_p = float('nan')
    return {
        "n_samples": int(len(df)),
        "pearson_r": float(r) if np.isfinite(r) else float('nan'),
        "pearson_p": float(p) if np.isfinite(p) else float('nan'),
        "ols_open_coef": ols_open_coef,
        "ols_open_p": ols_open_p,
        "ols2_open_coef": ols2_open_coef,
        "ols2_open_p": ols2_open_p
    }


def approach3_sliding_windows(bin_df: pd.DataFrame, latent: pd.Series, width: float = 0.10, min_n: int = 1):
    lo = float(latent.min())
    hi = float(latent.max())
    step = width
    rows = []
    start = lo
    # Ensure alignment
    df = bin_df.copy()
    df["LATENT"] = latent.values
    while start <= hi - width + 1e-9:
        mask = (df["LATENT"] >= start) & (df["LATENT"] < start + width)
        sub = df.loc[mask].drop(columns=["LATENT"])
        n = sub.shape[0]
        if n >= min_n and sub.shape[0] > 1:
            # filter words appearing in >=2 participants
            common_mask = (sub.sum(axis=0) >= 2)
            common_cols = sub.columns[common_mask]
            if len(common_cols) >= 3:
                X = sub[common_cols].values
                cos = cosine_similarity_from_binary(X)
                A = greedy_planar_backbone(cos)
                aspl = largest_cc_aspl_from_adj(A)
            else:
                aspl = np.nan
            mean_open = float(latent.loc[mask].mean()) if n > 0 else np.nan
            rows.append({"open": mean_open, "aspl": float(aspl) if np.isfinite(aspl) else np.nan, "n": int(n)})
        else:
            rows.append({"open": float(latent.loc[mask].mean()) if n > 0 else np.nan, "aspl": np.nan, "n": int(n)})
        start += step
    res = pd.DataFrame(rows).dropna(subset=["open"])  # keep rows with defined openness
    # correlations
    valid_all = res.dropna(subset=["aspl"])  # require aspl to correlate
    if len(valid_all) > 1:
        r_all, p_all = stats.pearsonr(valid_all["open"], valid_all["aspl"])
    else:
        r_all, p_all = (np.nan, np.nan)
    valid_n20 = res[(res["n"] > 20) & (~res["aspl"].isna())]
    if len(valid_n20) > 1:
        r_n20, p_n20 = stats.pearsonr(valid_n20["open"], valid_n20["aspl"])
    else:
        r_n20, p_n20 = (np.nan, np.nan)
    return {
        "n_bins": int(len(res)),
        "pearson_r_all": float(r_all) if np.isfinite(r_all) else float('nan'),
        "pearson_p_all": float(p_all) if np.isfinite(p_all) else float('nan'),
        "pearson_r_n_gt_20": float(r_n20) if np.isfinite(r_n20) else float('nan'),
        "pearson_p_n_gt_20": float(p_n20) if np.isfinite(p_n20) else float('nan')
    }


def main():
    flu, opn = load_data()
    # median split on LATENT
    if "LATENT" not in opn.columns:
        raise RuntimeError("LATENT column not found in openness data.")
    groups = opn["LATENT"] > opn["LATENT"].median()
    # Build binary matrix of words
    bin_df = build_binary_matrix(flu)
    # Approach 1: Paired t-test of ASPL between high vs low groups using 90% node bootstrap
    appr1 = approach1_group_bootstrap(bin_df, groups=groups, node_rate=0.9, n_boot=1000)
    # Approach 2: Participant bootstrap (n=20)
    appr2 = approach2_participant_bootstrap(bin_df, latent=opn["LATENT"], n_boot=1000, sub_n=20)
    # Approach 3: Sliding windows width=0.10
    appr3 = approach3_sliding_windows(bin_df, latent=opn["LATENT"], width=0.10, min_n=1)

    results = {
        "task_id": "Task1",
        "approach1": appr1,
        "approach2": appr2,
        "approach3": appr3
    }
    with open(os.path.join(ARTIFACTS_DIR, "task1_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    # also write execution_result.json for the orchestrator to find
    with open(os.path.join(ARTIFACTS_DIR, "execution_result.json"), "w") as f:
        json.dump({"Task1": results}, f, indent=2)

    print("Task1 completed. Results written to /app/artifacts/task1_results.json")


if __name__ == "__main__":
    main()
