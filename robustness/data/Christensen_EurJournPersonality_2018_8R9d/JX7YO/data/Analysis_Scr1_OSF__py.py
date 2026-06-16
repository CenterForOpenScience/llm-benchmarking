import json
import math
import itertools
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
import networkx as nx
from scipy import stats

# Utilities

def clean_word(x):
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in ('', 'nan', 'na', 'none', '-1', 'missing'):
        return None
    return s


def build_cooccurrence(word_sets, n_participants):
    """Build global co-occurrence counts for unordered word pairs.
    word_sets: list of sets of words per participant
    n_participants: total number of participants (denominator for p)
    Returns: dict mapping (w1, w2) -> count
    """
    counts = Counter()
    for ws in word_sets:
        if len(ws) < 2:
            continue
        # combinations on set avoids duplicate words within a participant
        for w1, w2 in itertools.combinations(sorted(ws), 2):
            counts[(w1, w2)] += 1
    return counts


def pair_entropy(count, n):
    # probability of co-occurrence across participants
    if n <= 0:
        return 0.0
    p = count / n
    # binary entropy in bits
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def participation_coefficients(G, partition):
    """Compute participation coefficient for each node in weighted graph G given a partition dict node->community.
    PC_i = 1 - sum_s (k_is / k_i)^2 where k_is is sum of weights of edges from i to nodes in community s, and k_i is total strength of i.
    """
    pcs = {}
    for n in G.nodes():
        k_i = 0.0
        comm_weights = defaultdict(float)
        for nbr, data in G[n].items():
            w = data.get('weight', 1.0)
            k_i += w
            s = partition.get(nbr, -1)
            comm_weights[s] += w
        if k_i <= 0:
            pcs[n] = 0.0
        else:
            frac_sq_sum = 0.0
            for s, wsum in comm_weights.items():
                frac = wsum / k_i
                frac_sq_sum += frac * frac
            pcs[n] = 1.0 - frac_sq_sum
    return pcs


def median_shortest_path_length(tree):
    # Compute all pairs shortest path lengths with weight
    nodes = list(tree.nodes())
    if len(nodes) < 2:
        return np.nan
    dists = []
    # Use all_pairs_dijkstra_path_length
    all_lengths = dict(nx.all_pairs_dijkstra_path_length(tree, weight='weight'))
    for i, u in enumerate(nodes):
        lu = all_lengths.get(u, {})
        for v in nodes[i+1:]:
            d = lu.get(v, np.nan)
            if not np.isnan(d):
                dists.append(d)
    if len(dists) == 0:
        return np.nan
    return float(np.median(dists))


def compute_metrics_for_participant(words_list, counts, n_participants):
    # words_list: list of strings (possibly with duplicates), we will treat unique set
    ws = sorted(set([w for w in words_list if w is not None]))
    if len(ws) < 2:
        return None
    # Build graphs: distance graph using entropy as distance; strength graph weight = 1/entropy
    Gd = nx.Graph()
    Gs = nx.Graph()
    Gd.add_nodes_from(ws)
    Gs.add_nodes_from(ws)
    for w1, w2 in itertools.combinations(ws, 2):
        key = (w1, w2) if (w1, w2) in counts else (w2, w1)
        c = counts.get(key, 0)
        H = pair_entropy(c, n_participants)
        # Replace zero-entropy edges with 0.99 as specified
        if H <= 0.0:
            H = 0.99
        Gd.add_edge(w1, w2, weight=H)
        # Strength graph: weight inversely proportional to entropy
        weight = 1.0 / H if H > 0 else 0.0
        Gs.add_edge(w1, w2, weight=weight)

    # Minimum Spanning Tree on distance graph
    try:
        mst = nx.minimum_spanning_tree(Gd, weight='weight')
    except Exception:
        mst = Gd.copy()

    cpl = median_shortest_path_length(mst)

    # Betweenness centrality on MST
    try:
        bc = nx.betweenness_centrality(mst, weight='weight', normalized=True)
        betw_mean = float(np.mean(list(bc.values()))) if len(bc) > 0 else np.nan
        betw_max = float(np.max(list(bc.values()))) if len(bc) > 0 else np.nan
    except Exception:
        betw_mean = np.nan
        betw_max = np.nan

    # Louvain partition on strength graph
    try:
        import community as community_louvain  # python-louvain
        partition = community_louvain.best_partition(Gs, weight='weight')
    except Exception:
        # Fallback: single community
        partition = {n: 0 for n in Gs.nodes()}

    pcs = participation_coefficients(Gs, partition)
    if len(pcs) == 0:
        part_mean = np.nan
    else:
        # Mode of PCs
        values = list(pcs.values())
        try:
            mode_val = stats.mode(values, nan_policy='omit', keepdims=True).mode
            if isinstance(mode_val, np.ndarray):
                mode_val = float(mode_val[0]) if len(mode_val) > 0 else np.nan
            else:
                mode_val = float(mode_val)
        except Exception:
            # Fallback: rounded mode via histogram
            hist, bin_edges = np.histogram(values, bins=10)
            mode_bin = np.argmax(hist)
            mode_val = float((bin_edges[mode_bin] + bin_edges[mode_bin+1]) / 2.0)
        sel = [v for v in values if (not np.isnan(v)) and (v > mode_val)]
        if len(sel) == 0:
            part_mean = float(np.nanmean(values))
        else:
            part_mean = float(np.mean(sel))

    return {
        'cpl': float(cpl) if not np.isnan(cpl) else np.nan,
        'betw_mean': betw_mean if not np.isnan(betw_mean) else np.nan,
        'betw_max': betw_max if not np.isnan(betw_max) else np.nan,
        'part_mean': part_mean if not np.isnan(part_mean) else np.nan
    }


def main():
    input_path = '/app/data/FINAL demo open fluency.csv'
    out_path = '/app/data/task1_results.json'
    df = pd.read_csv(input_path)

    # Build word lists per participant
    vf_cols = [c for c in df.columns if c.startswith('vf_an_')]
    words_per_row = []
    for idx, row in df[vf_cols].iterrows():
        ws = set()
        for c in vf_cols:
            w = clean_word(row[c])
            if w is not None:
                ws.add(w)
        words_per_row.append(ws)

    n_participants = df.shape[0]

    # Build global co-occurrence counts
    counts = build_cooccurrence(words_per_row, n_participants)

    # Compute openness average
    for col in ['o_ffi', 'oi_bfas', 'o_bfas', 'i_bfas']:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        else:
            df[col] = np.nan
    df['openness_average'] = df[['o_ffi', 'oi_bfas', 'o_bfas', 'i_bfas']].mean(axis=1, skipna=True)

    # Compute metrics per participant
    results = []
    for i, row in df.iterrows():
        words_list = []
        for c in vf_cols:
            words_list.append(clean_word(row[c]))
        metrics = compute_metrics_for_participant(words_list, counts, n_participants)
        if metrics is None:
            res = {'id': row.get('id', i), 'cpl': np.nan, 'betw_mean': np.nan, 'betw_max': np.nan, 'part_mean': np.nan}
        else:
            res = {'id': row.get('id', i)}
            res.update(metrics)
        res['openness_average'] = row['openness_average']
        results.append(res)

    res_df = pd.DataFrame(results)

    # Filter: valid cpl and values present; cpl < 10 should retain most/all rows per dataset
    filt_df = res_df[(~res_df['cpl'].isna()) & (res_df['cpl'] < 10) & (~res_df['part_mean'].isna()) & (~res_df['openness_average'].isna())].copy()

    # Save filtered data for verification
    try:
        filt_df.to_csv('/app/data/task1_filtered.csv', index=False)
    except Exception:
        pass

    # Spearman correlation between openness_average and part_mean
    corr_rho, corr_p, n_corr = (np.nan, np.nan, 0)
    try:
        tmp_corr = filt_df[['openness_average', 'part_mean']].dropna()
        if tmp_corr.shape[0] >= 3:
            corr_rho, corr_p = stats.spearmanr(tmp_corr['openness_average'], tmp_corr['part_mean'])
            n_corr = int(tmp_corr.shape[0])
    except Exception:
        pass

    # Median split and Welch t-test
    t_stat, t_p, n_low, n_high = (np.nan, np.nan, 0, 0)
    try:
        tmp_t = filt_df[['openness_average', 'part_mean']].dropna()
        if tmp_t.shape[0] >= 4:
            med = np.median(tmp_t['openness_average'])
            low = tmp_t[tmp_t['openness_average'] <= med]['part_mean']
            high = tmp_t[tmp_t['openness_average'] > med]['part_mean']
            n_low, n_high = int(low.shape[0]), int(high.shape[0])
            if n_low >= 2 and n_high >= 2:
                t_res = stats.ttest_ind(low, high, equal_var=False, nan_policy='omit')
                t_stat = float(t_res.statistic)
                t_p = float(t_res.pvalue)
    except Exception:
        pass

    output = {
        'task': 'Task1',
        'spearman': {
            'rho': None if (pd.isna(corr_rho)) else float(corr_rho),
            'p_value': None if (pd.isna(corr_p)) else float(corr_p),
            'n': int(n_corr)
        },
        'ttest_median_split': {
            't_stat': None if (pd.isna(t_stat)) else float(t_stat),
            'p_value': None if (pd.isna(t_p)) else float(t_p),
            'n_low': int(n_low),
            'n_high': int(n_high)
        }
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    # Also save per-participant metrics for transparency
    try:
        metrics_out = '/app/data/task1_participant_metrics.csv'
        res_df.to_csv(metrics_out, index=False)
    except Exception:
        pass

if __name__ == '__main__':
    main()
