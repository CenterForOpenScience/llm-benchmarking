import json
import os
import re
import sys
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

np.random.seed(12345)

DATA_PATH = "/app/data/FINAL demo open fluency.csv"
OUT_PATH = "/app/data/task2_results.json"

# ---------------------------
# Helpers (shared with Task1 logic, duplicated for self-containment)
# ---------------------------

def clean_token(x: str) -> str:
    if x is None:
        return ""
    if isinstance(x, float) and np.isnan(x):
        return ""
    s = str(x).strip().lower()
    s = re.sub(r"[^a-zA-Z\s-]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build_participant_tokens(df: pd.DataFrame, vf_cols) -> list:
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
    for c in cols:
        tmp[c] = pd.to_numeric(tmp[c], errors='coerce')
        if tmp[c].isna().any():
            tmp[c] = tmp[c].fillna(tmp[c].mean(skipna=True))
    zcols = [zscore_series(tmp[c]) for c in cols]
    zmat = np.vstack([z.values for z in zcols]).T
    return pd.Series(zmat.mean(axis=1), index=df.index)


def binary_matrix(words, participant_tokens):
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
    if X.shape[1] < 2 or X.shape[0] < 2:
        return np.zeros((X.shape[0], X.shape[0]), dtype=float)
    with np.errstate(invalid='ignore'):
        C = np.corrcoef(X)
    C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
    W = np.where(C > 0, C, 0.0)
    np.fill_diagonal(W, 0.0)
    return W


def graph_from_weights(words, W):
    G = nx.Graph()
    for w in words:
        G.add_node(w)
    V = len(words)
    for i in range(V):
        for j in range(i+1, V):
            w = W[i, j]
            if w > 0:
                dist = 1.0 / w if w > 0 else np.inf
                G.add_edge(words[i], words[j], weight=float(w), distance=float(dist))
    return G


def largest_component(G: nx.Graph):
    if G.number_of_nodes() == 0:
        return G
    comps = list(nx.connected_components(G))
    if not comps:
        return G
    biggest = max(comps, key=len)
    return G.subgraph(biggest).copy()


def compute_aspl(G: nx.Graph):
    if G.number_of_nodes() == 0 or G.number_of_edges() == 0:
        return np.nan
    H = largest_component(G)
    try:
        return float(nx.average_shortest_path_length(H, weight='distance')) if H.number_of_nodes() > 1 else np.nan
    except Exception:
        return np.nan


def ttest_summary(a, b):
    a = np.array(a, dtype=float)
    b = np.array(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        return {"t": np.nan, "p": np.nan, "n1": int(len(a)), "n2": int(len(b)), "d": np.nan}
    t, p = stats.ttest_ind(a, b, equal_var=False, nan_policy='omit')
    na, nb = len(a), len(b)
    sa2, sb2 = np.var(a, ddof=1), np.var(b, ddof=1)
    sp = np.sqrt(((na-1)*sa2 + (nb-1)*sb2) / (na+nb-2)) if (na+nb-2) > 0 else np.nan
    d = (np.mean(a) - np.mean(b)) / sp if sp and sp > 0 else np.nan
    return {"t": float(t), "p": float(p), "n1": int(na), "n2": int(nb), "d": float(d) if not np.isnan(d) else np.nan}


# ---------------------------
# Main analysis
# ---------------------------

def main():
    if not os.path.exists(DATA_PATH):
        print(f"Data file not found at {DATA_PATH}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(DATA_PATH)

    vf_cols = [c for c in df.columns if re.match(r"^vf_an_\d+$", c)]
    if not vf_cols:
        vf_cols = [c for c in df.columns if c.startswith('vf_an_')]

    comp = openness_composite(df)
    med = np.median(comp.values)
    group_high = comp >= med
    group_low = comp < med

    df_low = df[group_low].reset_index(drop=True)
    df_high = df[group_high].reset_index(drop=True)

    parts_low = build_participant_tokens(df_low, vf_cols)
    parts_high = build_participant_tokens(df_high, vf_cols)

    # Determine equated node set from all participants (not bootstrapped) with threshold 2 in each group
    counts_low = {}
    for toks in parts_low:
        for w in toks:
            counts_low[w] = counts_low.get(w, 0) + 1
    counts_high = {}
    for toks in parts_high:
        for w in toks:
            counts_high[w] = counts_high.get(w, 0) + 1
    nodes = sorted([w for w in counts_low.keys() if counts_low.get(w, 0) >= 2 and counts_high.get(w, 0) >= 2])

    if len(nodes) < 3:
        # Not enough nodes for network metrics
        results = {"error": "Too few equated nodes to build networks", "n_nodes": len(nodes)}
        with open(OUT_PATH, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote results to {OUT_PATH}")
        return

    # Build full graphs once (used to draw 90% subsets)
    X_low = binary_matrix(nodes, parts_low)
    X_high = binary_matrix(nodes, parts_high)
    W_low = corr_positive_weights(X_low)
    W_high = corr_positive_weights(X_high)
    G_low_full = graph_from_weights(nodes, W_low)
    G_high_full = graph_from_weights(nodes, W_high)

    n_nodes = len(nodes)
    retain = int(np.floor(0.9 * n_nodes))
    if retain < 2:
        retain = max(2, retain)

    n_boot = int(os.environ.get("N_BOOT", "1000"))
    aspl_low, aspl_high = [], []

    print(f"Starting node-wise bootstrap ({n_boot} iters) retaining 90% of {n_nodes} nodes -> {retain} nodes.")

    for b in range(n_boot):
        sel = np.random.choice(nodes, size=retain, replace=False)
        Hlow = G_low_full.subgraph(sel).copy()
        Hhigh = G_high_full.subgraph(sel).copy()
        aspl_low.append(compute_aspl(Hlow))
        aspl_high.append(compute_aspl(Hhigh))
        if (b + 1) % max(1, n_boot // 10) == 0:
            print(f"Completed {b+1}/{n_boot}")

    test = ttest_summary(aspl_low, aspl_high)

    results = {
        "n_boot": n_boot,
        "n_nodes": int(n_nodes),
        "retain_nodes": int(retain),
        "aspl_means": {"low": float(np.nanmean(aspl_low)), "high": float(np.nanmean(aspl_high))},
        "t_test": test
    }

    with open(OUT_PATH, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote results to {OUT_PATH}")


if __name__ == "__main__":
    main()
