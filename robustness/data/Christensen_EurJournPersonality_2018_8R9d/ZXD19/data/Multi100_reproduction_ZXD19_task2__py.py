import os
import json
import random
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

random.seed(42)
np.random.seed(42)

DATA_DIR = "/app/data"
ARTIFACTS_DIR = "/app/artifacts"
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# Utilities

def normalize_token(s: str) -> str:
    if s is None:
        return ""
    s = str(s).strip().lower()
    s = "".join([ch for ch in s if ch.isalpha()])
    if s in ("", "na", "nana", "nan", "none", "dontknow", "dk", "n", "idk"):
        return ""
    return s


def build_binary_matrix(flu: pd.DataFrame) -> pd.DataFrame:
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
    return pd.DataFrame(mat, columns=vocab)


def cosine_similarity_from_binary(X: np.ndarray) -> np.ndarray:
    Xt = X.T
    dots = Xt @ X
    norms = np.sqrt(np.diag(dots))
    denom = norms[:, None] * norms[None, :]
    with np.errstate(divide='ignore', invalid='ignore'):
        cos = np.true_divide(dots, denom)
        cos[~np.isfinite(cos)] = 0.0
    np.fill_diagonal(cos, 0.0)
    return cos


def greedy_planar_backbone(weights: np.ndarray) -> np.ndarray:
    n = weights.shape[0]
    G = nx.Graph()
    G.add_nodes_from(range(n))
    edges = []
    for i in range(n):
        for j in range(i+1, n):
            w = float(weights[i, j])
            if w > 0:
                edges.append((i, j, w))
    edges.sort(key=lambda x: x[2], reverse=True)
    max_edges = max(0, 3*n - 6)
    for (u, v, w) in edges:
        if G.number_of_edges() >= max_edges:
            break
        G.add_edge(u, v, weight=w)
        planar, _ = nx.check_planarity(G)
        if not planar:
            G.remove_edge(u, v)
    A = np.zeros((n, n), dtype=np.uint8)
    for u, v in G.edges():
        A[u, v] = 1
        A[v, u] = 1
    return A


def largest_cc_aspl_from_adj(A: np.ndarray) -> float:
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


def main():
    flu = pd.read_csv(os.path.join(DATA_DIR, "FINAL fluency.csv"))
    opn = pd.read_csv(os.path.join(DATA_DIR, "FINAL open.csv"))

    # group split
    groups = opn["LATENT"] > opn["LATENT"].median()

    # word binary matrix
    bin_df = build_binary_matrix(flu)

    # split
    bin1 = bin_df.loc[groups.values]
    bin2 = bin_df.loc[(~groups).values]

    # filter words appearing >=2 in each group
    mask1 = (bin1.sum(axis=0) >= 2)
    mask2 = (bin2.sum(axis=0) >= 2)
    common_cols = bin1.columns[mask1 & mask2]
    bin1c = bin1[common_cols]
    bin2c = bin2[common_cols]

    results = {
        "task_id": "Task2",
        "status": "insufficient_words"
    }

    if len(common_cols) >= 3:
        cos1 = cosine_similarity_from_binary(bin1c.values)
        cos2 = cosine_similarity_from_binary(bin2c.values)
        A1 = greedy_planar_backbone(cos1)
        A2 = greedy_planar_backbone(cos2)
        n_nodes = A1.shape[0]
        k = max(2, int(round(0.9 * n_nodes)))
        aspl1, aspl2 = [], []
        for i in range(1000):
            nodes = np.random.choice(n_nodes, size=k, replace=False)
            A1s = A1[np.ix_(nodes, nodes)]
            A2s = A2[np.ix_(nodes, nodes)]
            a1 = largest_cc_aspl_from_adj(A1s)
            a2 = largest_cc_aspl_from_adj(A2s)
            if np.isfinite(a1) and np.isfinite(a2):
                aspl1.append(a1)
                aspl2.append(a2)
        if len(aspl1) > 1:
            t_stat, p_val = stats.ttest_rel(aspl1, aspl2, nan_policy='omit')
            results = {
                "task_id": "Task2",
                "status": "ok",
                "mean_aspl_high": float(np.mean(aspl1)),
                "mean_aspl_low": float(np.mean(aspl2)),
                "t_stat": float(t_stat),
                "p_value": float(p_val),
                "n_nodes": int(n_nodes)
            }
        else:
            results = {
                "task_id": "Task2",
                "status": "insufficient_aspl",
                "n_nodes": int(n_nodes)
            }

    with open(os.path.join(ARTIFACTS_DIR, "task2_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    # also write execution_result.json to consolidate
    exec_path = os.path.join(ARTIFACTS_DIR, "execution_result.json")
    data = {}
    if os.path.exists(exec_path):
        try:
            data = json.load(open(exec_path, 'r'))
        except Exception:
            data = {}
    data["Task2"] = results
    with open(exec_path, 'w') as f:
        json.dump(data, f, indent=2)

    print("Task2 completed. Results written to /app/artifacts/task2_results.json")


if __name__ == "__main__":
    main()
