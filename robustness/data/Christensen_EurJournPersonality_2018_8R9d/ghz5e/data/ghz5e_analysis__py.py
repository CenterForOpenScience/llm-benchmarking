import argparse
import json
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
try:
    import community as community_louvain
except Exception:
    community_louvain = None


def log(msg):
    print(msg, file=sys.stderr, flush=True)


def clean_token(x: str) -> str:
    if not isinstance(x, str):
        return ''
    s = x.strip().lower()
    # remove surrounding punctuation and extra spaces
    s = ''.join([ch if ch.isalpha() or ch == ' ' else ' ' for ch in s])
    s = ' '.join(s.split())
    # simple singularization: remove trailing 's' for tokens > 3 chars
    if len(s) > 3 and s.endswith('s'):
        s = s[:-1]
    return s


def load_and_merge(data_file: str, open_file: str = None, id_col: str = 'id', openness_col: str = 'LATENT'):
    df_flu = pd.read_csv(data_file)
    if id_col not in df_flu.columns:
        raise ValueError(f"id_col '{id_col}' not found in fluency data.")
    # Try to locate openness column in provided file; otherwise merge with open_file or default
    if openness_col in df_flu.columns:
        df = df_flu.copy()
    else:
        if open_file is None:
            # default path inside container
            default_open = '/app/data/FINAL open.csv'
            if os.path.exists(default_open):
                open_file = default_open
            else:
                raise ValueError("Openness column not found in fluency file and open file not provided nor available at /app/data/FINAL open.csv")
        df_open = pd.read_csv(open_file)
        if id_col not in df_open.columns:
            raise ValueError(f"id_col '{id_col}' not found in openness file.")
        if openness_col not in df_open.columns:
            # try a few alternatives
            for cand in ['LATENT', 'LATENT2', 'AVERAGE', 'o_ffi', 'o_bfas']:
                if cand in df_open.columns:
                    openness_col = cand
                    log(f"Using alternative openness_col '{cand}' from openness file.")
                    break
        if openness_col not in df_open.columns:
            raise ValueError(f"openness_col '{openness_col}' not found in openness file.")
        df = pd.merge(df_flu, df_open[[id_col, openness_col]], on=id_col, how='left')
        if df[openness_col].isna().any():
            nmiss = int(df[openness_col].isna().sum())
            log(f"Warning: {nmiss} rows missing openness scores after merge; they will be dropped.")
            df = df.dropna(subset=[openness_col])
    return df, openness_col


def get_response_columns(df: pd.DataFrame, response_prefix: str):
    cols = [c for c in df.columns if c.startswith(response_prefix)]
    if not cols:
        # try uppercase/lowercase variants
        pref = response_prefix.lower()
        cols = [c for c in df.columns if c.lower().startswith(pref)]
    if not cols:
        raise ValueError(f"No response columns found with prefix '{response_prefix}'.")
    return cols


def build_binary_matrix(df: pd.DataFrame, response_cols):
    # Clean and collect tokens per participant
    cleaned = df[response_cols].astype(str).applymap(clean_token)
    # Build vocabulary
    # Flatten values and collect unique non-empty tokens
    tokens = pd.unique(cleaned.values.ravel())
    vocab = [t for t in tokens if isinstance(t, str) and t != '']
    vocab = sorted(set(vocab))
    if len(vocab) == 0:
        raise ValueError("No valid tokens found in responses after cleaning.")
    # Map tokens to column indices
    tok_to_idx = {t: i for i, t in enumerate(vocab)}
    # Initialize binary matrix
    n = cleaned.shape[0]
    m = len(vocab)
    X = np.zeros((n, m), dtype=np.uint8)
    for i in range(n):
        row = cleaned.iloc[i]
        seen = set()
        for val in row.values:
            if val:
                seen.add(val)
        for t in seen:
            j = tok_to_idx.get(t)
            if j is not None:
                X[i, j] = 1
    mat = pd.DataFrame(X, index=df.index, columns=vocab)
    return mat


def construct_graph_from_matrix(mat_nodes_by_participants: pd.DataFrame):
    # mat is participants x nodes; we need nodes x participants for similarity
    if mat_nodes_by_participants.shape[1] < 2:
        # not enough nodes
        G = nx.Graph()
        for c in mat_nodes_by_participants.columns:
            G.add_node(c)
        return G
    A = cosine_similarity(mat_nodes_by_participants.values.T)
    np.fill_diagonal(A, 0.0)
    # Build graph with edges where similarity > 0
    G = nx.Graph()
    nodes = list(mat_nodes_by_participants.columns)
    G.add_nodes_from(nodes)
    # add edges for positive similarities
    rows, cols = np.where(A > 0)
    for i, j in zip(rows, cols):
        if i < j:
            w = float(A[i, j])
            if w > 0:
                G.add_edge(nodes[i], nodes[j], weight=w)
    return G


def average_shortest_path_length_weighted(G: nx.Graph):
    if G.number_of_nodes() < 2 or G.number_of_edges() == 0:
        return float('nan')
    # Work on largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    H = G.subgraph(largest_cc).copy()
    if H.number_of_nodes() < 2 or H.number_of_edges() == 0:
        return float('nan')
    # define distance as inverse of weight
    for u, v, d in H.edges(data=True):
        w = d.get('weight', 0.0)
        if w <= 0:
            # If no weight, set a large distance
            d['distance'] = 1e9
        else:
            d['distance'] = 1.0 / w
    try:
        aspl = nx.average_shortest_path_length(H, weight='distance')
    except Exception:
        aspl = float('nan')
    return float(aspl)


def compute_measures(G: nx.Graph):
    # ASPL
    aspl = average_shortest_path_length_weighted(G)
    # Clustering coefficient (weighted if weights present)
    try:
        cc = nx.average_clustering(G, weight='weight')
    except Exception:
        cc = float('nan')
    # Modularity Q using Louvain if available
    q = float('nan')
    if community_louvain is not None and G.number_of_nodes() >= 2 and G.number_of_edges() > 0:
        try:
            part = community_louvain.best_partition(G, weight='weight')
            q = community_louvain.modularity(part, G, weight='weight')
        except Exception:
            q = float('nan')
    return {'ASPL': float(aspl), 'CC': float(cc), 'Q': float(q)}


def bootstrap_differences(mat_low: pd.DataFrame, mat_high: pd.DataFrame, props, iters=200, random_state=123):
    rng = np.random.default_rng(random_state)
    nodes = list(mat_low.columns)
    results = {}
    for p in props:
        diffs = {'ASPL': [], 'CC': [], 'Q': []}
        k = max(2, int(round(len(nodes) * float(p))))
        for b in range(iters):
            subset = rng.choice(nodes, size=k, replace=False)
            mL = mat_low[subset]
            mH = mat_high[subset]
            GL = construct_graph_from_matrix(mL)
            GH = construct_graph_from_matrix(mH)
            mL_meas = compute_measures(GL)
            mH_meas = compute_measures(GH)
            for key in diffs.keys():
                dval = (mH_meas.get(key, np.nan) - mL_meas.get(key, np.nan))
                diffs[key].append(dval)
        # summarize
        summ = {}
        for key, arr in diffs.items():
            a = np.array(arr, dtype=float)
            mean = float(np.nanmean(a))
            sd = float(np.nanstd(a, ddof=1)) if len(a) > 1 else float('nan')
            t_like = float('nan')
            if np.isfinite(sd) and sd > 0 and len(a) > 1:
                t_like = mean / (sd / np.sqrt(len(a)))
            summ[key] = {
                'mean_diff_H_minus_L': mean,
                'sd': sd,
                't_like': t_like,
                'n_bootstrap': int(len(a)),
                'prop_nodes_retained': float(p)
            }
        results[str(p)] = summ
    return results


def run_analysis(task: str, data_file: str, id_col: str, openness_col: str, response_prefix: str, open_file: str = None, bootstrap_prop: float = None, boot_iters: int = 200):
    # Load and merge
    log(f"Loading data_file={data_file} open_file={open_file} id_col={id_col} openness_col={openness_col}")
    df, openness_col = load_and_merge(data_file, open_file=open_file, id_col=id_col, openness_col=openness_col)
    # Select response columns
    resp_cols = get_response_columns(df, response_prefix)
    # Build binary matrix of responses
    mat_all = build_binary_matrix(df, resp_cols)
    # Split into High vs Low openness by sorted halves
    df2 = df[[id_col, openness_col]].copy()
    df2 = df2.sort_values(by=openness_col, ascending=True).reset_index(drop=True)
    n = df2.shape[0]
    n_low = n // 2
    low_ids = set(df2.iloc[:n_low][id_col].tolist())
    high_ids = set(df2.iloc[n_low:][id_col].tolist())
    # Align mat_all index to df id ordering
    if id_col in df.index.names or id_col in df.columns:
        mat_all = mat_all.set_index(df[id_col].values)
    # Subset matrices for groups
    mat_low = mat_all.loc[mat_all.index.isin(low_ids)].copy()
    mat_high = mat_all.loc[mat_all.index.isin(high_ids)].copy()
    # Remove nodes endorsed by fewer than 2 participants per group
    low_keep = mat_low.columns[(mat_low.sum(axis=0) >= 2)]
    high_keep = mat_high.columns[(mat_high.sum(axis=0) >= 2)]
    keep_nodes = sorted(list(set(low_keep).intersection(set(high_keep))))
    # Equate nodes across groups
    mat_low_eq = mat_low[keep_nodes]
    mat_high_eq = mat_high[keep_nodes]
    # Compute full-network graphs and measures
    GL_full = construct_graph_from_matrix(mat_low_eq)
    GH_full = construct_graph_from_matrix(mat_high_eq)
    meas_low = compute_measures(GL_full)
    meas_high = compute_measures(GH_full)
    # Determine bootstrap proportions
    if task == 'Task1' and bootstrap_prop is None:
        props = [0.5, 0.6, 0.7, 0.8, 0.9]
    elif task == 'Task2' and bootstrap_prop is None:
        props = [0.9]
    else:
        if bootstrap_prop is None:
            props = [0.9]
        else:
            props = [float(bootstrap_prop)]
    boot_summary = bootstrap_differences(mat_low_eq, mat_high_eq, props=props, iters=boot_iters, random_state=123)
    # Prepare result
    result = {
        'task': task,
        'n_total': int(n),
        'n_low': int(mat_low.shape[0]),
        'n_high': int(mat_high.shape[0]),
        'n_nodes_equated': int(len(keep_nodes)),
        'group_measures': {
            'Low': meas_low,
            'High': meas_high
        },
        'bootstrap': boot_summary,
        'notes': {
            'id_col': id_col,
            'openness_col': openness_col,
            'response_prefix': response_prefix
        }
    }
    print(json.dumps(result, indent=2))


def run_capture(func):
    from io import StringIO
    import contextlib
    buf = StringIO()
    with contextlib.redirect_stdout(buf):
        func()
    return buf.getvalue()


def main():
    parser = argparse.ArgumentParser(description='GHZ5E semantic network analysis (Python)')
    parser.add_argument('--task', type=str, default=os.environ.get('PR_TASK', 'BOTH'))
    parser.add_argument('--data_file', type=str, default=os.environ.get('PR_DATA_FILE', '/app/data/FINAL fluency.csv'))
    parser.add_argument('--open_file', type=str, default=os.environ.get('PR_OPEN_FILE', '/app/data/FINAL open.csv'))
    parser.add_argument('--id_col', type=str, default=os.environ.get('PR_ID_COL', 'id'))
    parser.add_argument('--openness_col', type=str, default=os.environ.get('PR_OPENNESS_COL', 'LATENT'))
    parser.add_argument('--response_prefix', type=str, default=os.environ.get('PR_RESPONSE_PREFIX', 'vf_an_'))
    parser.add_argument('--bootstrap_prop', type=float, default=None)
    parser.add_argument('--boot_iters', type=int, default=int(os.environ.get('PR_BOOT_ITERS', '200')))

    args = parser.parse_args()

    try:
        task = (args.task or '').strip()
        if task.upper() in ('BOTH', 'ALL') or task == '':
            # Run Task1 and Task2 and emit combined JSON
            res1 = json.loads(
                run_capture(lambda: run_analysis(task='Task1',
                                  data_file=args.data_file,
                                  id_col=args.id_col,
                                  openness_col=args.openness_col,
                                  response_prefix=args.response_prefix,
                                  open_file=args.open_file,
                                  bootstrap_prop=None,
                                  boot_iters=args.boot_iters))
            )
            res2 = json.loads(
                run_capture(lambda: run_analysis(task='Task2',
                                  data_file=args.data_file,
                                  id_col=args.id_col,
                                  openness_col=args.openness_col,
                                  response_prefix=args.response_prefix,
                                  open_file=args.open_file,
                                  bootstrap_prop=0.9,
                                  boot_iters=args.boot_iters))
            )
            combined = { 'Task1': res1, 'Task2': res2 }
            try:
                with open('/app/data/execution_result.json', 'w') as f:
                    json.dump(combined, f, indent=2)
            except Exception as e:
                log(f"Warning: failed to write execution_result.json: {e}")
            print(json.dumps(combined, indent=2))
        else:
            # Single-task mode
            run_analysis(task=task,
                         data_file=args.data_file,
                         id_col=args.id_col,
                         openness_col=args.openness_col,
                         response_prefix=args.response_prefix,
                         open_file=args.open_file,
                         bootstrap_prop=args.bootstrap_prop,
                         boot_iters=args.boot_iters)
    except Exception as e:
        log(f"ERROR: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()
