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
    counts = Counter()
    for ws in word_sets:
        if len(ws) < 2:
            continue
        for w1, w2 in itertools.combinations(sorted(ws), 2):
            counts[(w1, w2)] += 1
    return counts


def pair_entropy(count, n):
    if n <= 0:
        return 0.0
    p = count / n
    if p <= 0.0 or p >= 1.0:
        return 0.0
    return -(p * math.log2(p) + (1 - p) * math.log2(1 - p))


def participation_coefficients(G, partition):
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
    nodes = list(tree.nodes())
    if len(nodes) < 2:
        return np.nan
    dists = []
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


def compute_metrics_for_participant(words_list, counts, n_participants, keep_ratio=0.9):
    ws = sorted(set([w for w in words_list if w is not None]))
    if len(ws) < 2:
        return None
    Gd = nx.Graph()
    Gs = nx.Graph()
    Gd.add_nodes_from(ws)
    Gs.add_nodes_from(ws)
    for w1, w2 in itertools.combinations(ws, 2):
        key = (w1, w2) if (w1, w2) in counts else (w2, w1)
        c = counts.get(key, 0)
        H = pair_entropy(c, n_participants)
        if H <= 0.0:
            H = 0.99
        Gd.add_edge(w1, w2, weight=H)
        weight = 1.0 / H if H > 0 else 0.0
        Gs.add_edge(w1, w2, weight=weight)

    # Remove top 10% by eigenvector centrality (keep 90%)
    try:
        ev = nx.eigenvector_centrality_numpy(Gs, weight='weight')
        n_keep = max(1, int(math.ceil(len(ev) * keep_ratio)))
        # Highest centrality nodes removed so we keep lower 90%
        nodes_sorted = sorted(ev.items(), key=lambda x: x[1], reverse=True)
        to_remove = [n for n, _ in nodes_sorted[n_keep:]]
        Gd.remove_nodes_from(to_remove)
        Gs.remove_nodes_from(to_remove)
    except Exception:
        pass

    if Gd.number_of_nodes() < 2:
        return None

    try:
        mst = nx.minimum_spanning_tree(Gd, weight='weight')
    except Exception:
        mst = Gd.copy()

    cpl = median_shortest_path_length(mst)

    try:
        bc = nx.betweenness_centrality(mst, weight='weight', normalized=True)
        betw_mean = float(np.mean(list(bc.values()))) if len(bc) > 0 else np.nan
        betw_max = float(np.max(list(bc.values()))) if len(bc) > 0 else np.nan
    except Exception:
        betw_mean = np.nan
        betw_max = np.nan

    try:
        import community as community_louvain
        partition = community_louvain.best_partition(Gs, weight='weight')
    except Exception:
        partition = {n: 0 for n in Gs.nodes()}

    pcs = participation_coefficients(Gs, partition)
    if len(pcs) == 0:
        part_mean = np.nan
    else:
        values = list(pcs.values())
        try:
            mode_val = stats.mode(values, nan_policy='omit', keepdims=True).mode
            if isinstance(mode_val, np.ndarray):
                mode_val = float(mode_val[0]) if len(mode_val) > 0 else np.nan
            else:
                mode_val = float(mode_val)
        except Exception:
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
    out_path = '/app/data/task2_results.json'
    df = pd.read_csv(input_path)

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
    counts = build_cooccurrence(words_per_row, n_participants)

    # Construct metrics with 90% node retention per subject
    results = []
    for i, row in df.iterrows():
        words_list = []
        for c in vf_cols:
            words_list.append(clean_word(row[c]))
        metrics = compute_metrics_for_participant(words_list, counts, n_participants, keep_ratio=0.9)
        if metrics is None:
            res = {'id': row.get('id', i), 'cpl': np.nan, 'betw_mean': np.nan, 'betw_max': np.nan, 'part_mean': np.nan}
        else:
            res = {'id': row.get('id', i)}
            res.update(metrics)
        # carry demographics for filtering/grouping
        res['d_gender'] = str(row.get('d_gender', '')).strip()
        results.append(res)

    res_df = pd.DataFrame(results)

    # Apply filters per plan, treating blanks as missing and excluding '-1'
    valid_gender = res_df['d_gender'].isin({'0', '1'})
    filt_df = res_df[(~res_df['cpl'].isna()) & (res_df['cpl'] < 10) & (~res_df['part_mean'].isna()) & valid_gender]

    # Welch t-test of part_mean by gender (0 vs 1)
    t_stat, p_value, n_g0, n_g1 = (np.nan, np.nan, 0, 0)
    excluded_blank = int((~valid_gender).sum())
    try:
        v0 = filt_df[filt_df['d_gender'] == '0']['part_mean']
        v1 = filt_df[filt_df['d_gender'] == '1']['part_mean']
        n_g0, n_g1 = int(v0.shape[0]), int(v1.shape[0])
        if n_g0 >= 2 and n_g1 >= 2:
            t_res = stats.ttest_ind(v0, v1, equal_var=False, nan_policy='omit')
            t_stat = float(t_res.statistic)
            p_value = float(t_res.pvalue)
    except Exception:
        pass

    output = {
        'task': 'Task2',
        'welch_ttest_by_gender': {
            't_stat': None if (pd.isna(t_stat)) else float(t_stat),
            'p_value': None if (pd.isna(p_value)) else float(p_value),
            'groups_compared': ['0', '1'],
            'n_group0': int(n_g0),
            'n_group1': int(n_g1),
            'n_excluded_nonbinary_or_blank': int(excluded_blank)
        }
    }

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)

    try:
        metrics_out = '/app/data/task2_participant_metrics.csv'
        res_df.to_csv(metrics_out, index=False)
    except Exception:
        pass

if __name__ == '__main__':
    main()
