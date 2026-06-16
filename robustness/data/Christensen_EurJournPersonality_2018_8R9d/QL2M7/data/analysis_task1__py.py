import json
import os
import re
import sys
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

# Modularity (Louvain)
try:
    import community as community_louvain  # python-louvain
except Exception:
    community_louvain = None

np.random.seed(12345)

DATA_PATH = "/app/data/FINAL demo open fluency.csv"
OUT_PATH = "/app/data/task1_results.json"

# ---------------------------
# Helpers
# ---------------------------

def clean_token(x: str) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip().lower()
    # remove punctuation and extra spaces
    s = re.sub(r"[^a-zA-Z\s-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_participant_tokens(df: pd.DataFrame, vf_cols) -> list:
    """Return list of sets; each element is the cleaned set of tokens for a participant."""
    parts = []
    for _, row in df.iterrows():
        toks = set()
        for c in vf_cols:
            tok = clean_token(row.get(c, ""))
            if tok:
                toks.add(tok)
        parts.append(toks)
    return parts


def zscore_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors='coerce')
    m = s.mean(skipna=True)
    sd = s.std(ddof=0, skipna=True)
    if pd.isna(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - m) / sd


def openness_composite(df: pd.DataFrame) -> pd.Series:
    cols = [c for c in ["o_ffi", "o_bfas", "o_neo"] if c in df.columns]
    if not cols:
        raise ValueError("No openness total columns found (expected any of o_ffi, o_bfas, o_neo)")
    tmp = df[cols].copy()
    # coerce to numeric and mean-impute per column
    for c in cols:
        tmp[c] = pd.to_numeric(tmp[c], errors='coerce')
        if tmp[c].isna().any():
            tmp[c] = tmp[c].fillna(tmp[c].mean(skipna=True))
    # z-score each then average
    zcols = [zscore_series(tmp[c]) for c in cols]
    zmat = np.vstack([z.values for z in zcols]).T
    comp = pd.Series(zmat.mean(axis=1), index=df.index)
    return comp


def binary_matrix(words, participant_tokens):
    """Build a binary matrix of shape (len(words), len(participants))."""
    word_index = {w: i for i, w in enumerate(words)}
    V = len(words)
    N = len(participant_tokens)
    X = np.zeros((V, N), dtype=float)
    for j, toks in enumerate(participant_tokens):
        if not toks:
            continue
        for w in toks:
            i = word_index.get(w)
            if i is not None:
                X[i, j] = 1.0
    return X


def corr_positive_weights(X):
    """Return positive-only Pearson correlation weights (VxV) from binary matrix (VxN)."""
    if X.shape[1] < 2 or X.shape[0] < 2:
        return np.zeros((X.shape[0], X.shape[0]), dtype=float)
    # Compute correlation across participants for word rows
    with np.errstate(invalid='ignore'):
        C = np.corrcoef(X)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    # Keep positive, zero out diagonal
    W = np.where(C > 0, C, 0.0)
    np.fill_diagonal(W, 0.0)
    return W


def graph_from_weights(words, W):
    G = nx.Graph()
    for w in words:
        G.add_node(w)
    V = len(words)
    for i in range(V):
        wi = W[i]
        for j in range(i+1, V):
            w = wi[j]
            if w > 0:
                # store weight and distance
                dist = 1.0 / w if w > 0 else np.inf
                G.add_edge(words[i], words[j], weight=float(w), distance=float(dist))
    return G


def largest_component(G: nx.Graph):
    if G.number_of_nodes() == 0:
        return G
    if nx.is_empty(G):
        return G
    comps = list(nx.connected_components(G))
    if not comps:
        return G
    biggest = max(comps, key=len)
    return G.subgraph(biggest).copy()


def compute_metrics(G: nx.Graph):
    # Use largest connected component for ASPL
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return {"ASPL": np.nan, "CC": np.nan, "Q": np.nan}
    H = largest_component(G)
    try:
        aspl = nx.average_shortest_path_length(H, weight='distance') if H.number_of_nodes() > 1 else np.nan
    except Exception:
        aspl = np.nan
    try:
        cc = nx.average_clustering(G, weight='weight')
    except Exception:
        cc = np.nan
    q = np.nan
    if community_louvain is not None and G.number_of_nodes() >= 2:
        try:
            part = community_louvain.best_partition(G, weight='weight')
            q = community_louvain.modularity(part, G, weight='weight')
        except Exception:
            q = np.nan
    return {"ASPL": float(aspl) if aspl is not None else np.nan,
            "CC": float(cc) if cc is not None else np.nan,
            "Q": float(q) if q is not None else np.nan}


def ttest_summary(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return {"t": np.nan, "p": np.nan, "n1": int(len(a)), "n2": int(len(b))}
    t, p = stats.ttest_ind(a, b, equal_var=False, nan_policy='omit')
    return {"t": float(t), "p": float(p), "n1": int(len(a)), "n2": int(len(b))}


# ---------------------------
# Main analysis
# ---------------------------

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at {DATA_PATH}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(DATA_PATH)

    # Identify fluency columns
    vf_cols = [c for c in df.columns if re.match(r"^vf_an_\d+$", c)]
    if not vf_cols:
        # fallback exact known columns
        vf_cols = [c for c in df.columns if c.startswith('vf_an_')]
    # Build openness composite and groups
    comp = openness_composite(df)
    med = np.median(comp.values)
    group_high = comp >= med
    group_low = comp < med

    df_low = df[group_low].reset_index(drop=True)
    df_high = df[group_high].reset_index(drop=True)

    parts_low = build_participant_tokens(df_low, vf_cols)
    parts_high = build_participant_tokens(df_high, vf_cols)

    n_low = len(parts_low)
    n_high = len(parts_high)

    # Case-wise bootstrap
    n_boot = int(os.environ.get("N_BOOT", "1000"))
    thresh = 2  # min producers per group for equated node set

    metrics_low = {"ASPL": [], "CC": [], "Q": []}
    metrics_high = {"ASPL": [], "CC": [], "Q": []}

    idx_low = np.arange(n_low)
    idx_high = np.arange(n_high)

    print(f"Starting case-wise bootstrap with {n_boot} iterations...", flush=True)

    for b in range(n_boot):
        # Resample participants within each group (with replacement)
        boot_low_idx = np.random.choice(idx_low, size=n_low, replace=True)
        boot_high_idx = np.random.choice(idx_high, size=n_high, replace=True)

        boot_low_parts = [parts_low[i] for i in boot_low_idx]
        boot_high_parts = [parts_high[i] for i in boot_high_idx]

        # Word counts per group
        counts_low = {}
        for toks in boot_low_parts:
            for w in toks:
                counts_low[w] = counts_low.get(w, 0) + 1
        counts_high = {}
        for toks in boot_high_parts:
            for w in toks:
                counts_high[w] = counts_high.get(w, 0) + 1
        # Equated nodes: appear at least thresh times in both groups
        nodes = sorted([w for w in counts_low.keys() if counts_low.get(w, 0) >= thresh and counts_high.get(w, 0) >= thresh])

        if len(nodes) < 3:
            # too few nodes; record NaNs
            for k in metrics_low:
                metrics_low[k].append(np.nan)
                metrics_high[k].append(np.nan)
            continue

        # Build binary matrices restricted to nodes
        X_low = binary_matrix(nodes, boot_low_parts)
        X_high = binary_matrix(nodes, boot_high_parts)

        # Correlation weights
        W_low = corr_positive_weights(X_low)
        W_high = corr_positive_weights(X_high)

        # Graphs and metrics
        G_low = graph_from_weights(nodes, W_low)
        G_high = graph_from_weights(nodes, W_high)

        m_low = compute_metrics(G_low)
        m_high = compute_metrics(G_high)

        for k in metrics_low:
            metrics_low[k].append(m_low[k])
            metrics_high[k].append(m_high[k])

        if (b + 1) % max(1, n_boot // 10) == 0:
            print(f"Completed {b+1}/{n_boot} bootstraps", flush=True)

    # Summaries and tests
    results = {
        "n_boot": n_boot,
        "group_sizes": {"low": int(n_low), "high": int(n_high)},
        "metric_means": {
            m: {"low": float(np.nanmean(metrics_low[m])) if len(metrics_low[m]) else np.nan,
                "high": float(np.nanmean(metrics_high[m])) if len(metrics_high[m]) else np.nan}
            for m in ["ASPL", "CC", "Q"]
        },
        "t_tests": {
            m: ttest_summary(metrics_low[m], metrics_high[m]) for m in ["ASPL", "CC", "Q"]
        }
    }

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote results to {OUT_PATH}")


if __name__ == "__main__":
    main()
