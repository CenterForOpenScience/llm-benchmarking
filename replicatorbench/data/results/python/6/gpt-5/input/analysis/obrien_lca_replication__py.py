import argparse, json, os, sys, math
import numpy as np
import pandas as pd
from scipy import stats
import pyreadstat

# ------------- Utility: Weighted LCA (categorical, local independence) -------------
class WeightedLCA:
    def __init__(self, n_classes=3, max_iter=200, tol=1e-6, n_starts=5, random_state=12345, smooth=1e-6):
        self.n_classes = n_classes
        self.max_iter = max_iter
        self.tol = tol
        self.n_starts = n_starts
        self.random_state = random_state
        self.smooth = smooth
        self.fitted_ = False

    def _preprocess(self, X):
        # X: DataFrame of categorical indicators with ints.
        # Map each column categories to 0..K-1
        self.col_maps_ = {}
        self.col_levels_ = {}
        X_mapped = pd.DataFrame(index=X.index)
        for col in X.columns:
            vals = pd.Categorical(X[col])
            cats = list(vals.categories)
            # Map NaN to NaN, categories to 0..K-1
            mapping = {cat: i for i, cat in enumerate(cats)}
            self.col_maps_[col] = mapping
            self.col_levels_[col] = len(cats)
            X_mapped[col] = X[col].map(mapping)
        return X_mapped.astype('float')  # keep NaNs as float

    def _init_params(self, X, rng):
        C = self.n_classes
        J = X.shape[1]
        # pi: (C,)
        pi = rng.random(C)
        pi = pi / pi.sum()
        # theta: list of length J; each is (C, Kj)
        theta = []
        for j, col in enumerate(X.columns):
            Kj = self.col_levels_[col]
            t = rng.random((C, Kj)) + self.smooth
            t = t / t.sum(axis=1, keepdims=True)
            theta.append(t)
        return pi, theta

    def _log_prob_x_given_c(self, Xi, theta):
        # Xi: array of length J with category indices (floats; NaN if missing)
        # Return log P(x_i | c) for each class c
        C = self.n_classes
        logp = np.zeros(C)
        for c in range(C):
            s = 0.0
            for j, xij in enumerate(Xi):
                if not np.isnan(xij):
                    x = int(xij)
                    p = theta[j][c, x]
                    if p <= 0:
                        p = self.smooth
                    s += math.log(p)
            logp[c] = s
        return logp

    def fit(self, X, weights=None):
        rng = np.random.default_rng(self.random_state)
        X = X.copy()
        Xp = self._preprocess(X)
        n, J = Xp.shape
        w = np.ones(n) if weights is None else np.asarray(weights).reshape(-1)
        w = np.where(np.isfinite(w), w, 0.0)
        best_ll = -np.inf
        best_params = None

        for start in range(self.n_starts):
            pi, theta = self._init_params(Xp, rng)
            prev_ll = -np.inf
            for it in range(self.max_iter):
                # E-step: responsibilities gamma (n, C)
                log_resp = np.zeros((n, self.n_classes))
                for i in range(n):
                    logp_x_c = self._log_prob_x_given_c(Xp.iloc[i].to_numpy(), theta)
                    # add log pi
                    logp = logp_x_c + np.log(pi + self.smooth)
                    # stabilize
                    m = np.max(logp)
                    logp -= m
                    p = np.exp(logp)
                    p_sum = p.sum()
                    if p_sum == 0:
                        p = np.ones_like(p) / len(p)
                    else:
                        p = p / p_sum
                    log_resp[i, :] = np.log(p + self.smooth)
                resp = np.exp(log_resp)

                # M-step with weights
                # Update pi
                wresp = w[:, None] * resp
                pi = wresp.sum(axis=0)
                if pi.sum() == 0:
                    pi = np.ones_like(pi) / len(pi)
                else:
                    pi = pi / pi.sum()

                # Update theta per variable
                new_theta = []
                for j, col in enumerate(Xp.columns):
                    Kj = self.col_levels_[col]
                    counts = np.zeros((self.n_classes, Kj)) + self.smooth
                    for k in range(Kj):
                        mask = (Xp[col].to_numpy() == k)
                        wk = w[mask][:, None] * resp[mask, :]
                        counts[:, k] += wk.sum(axis=0)
                    counts_sum = counts.sum(axis=1, keepdims=True)
                    counts_sum[counts_sum == 0] = 1.0
                    t = counts / counts_sum
                    new_theta.append(t)
                theta = new_theta

                # Compute weighted log-likelihood
                ll = 0.0
                for i in range(n):
                    logp_x_c = self._log_prob_x_given_c(Xp.iloc[i].to_numpy(), theta) + np.log(pi + self.smooth)
                    m = np.max(logp_x_c)
                    ll_i = m + np.log(np.exp(logp_x_c - m).sum())
                    ll += w[i] * ll_i

                if abs(ll - prev_ll) < self.tol:
                    break
                prev_ll = ll

            if ll > best_ll:
                best_ll = ll
                best_params = (pi.copy(), [t.copy() for t in theta])

        self.pi_, self.theta_ = best_params
        self.columns_ = list(X.columns)
        self.fitted_ = True
        self.loglik_ = best_ll
        return self

    def predict_proba(self, X):
        assert self.fitted_, "Model not fitted"
        # Map using stored mappings
        Xp = pd.DataFrame(index=X.index)
        for col in self.columns_:
            mapping = self.col_maps_[col]
            Xp[col] = X[col].map(mapping)
        Xp = Xp.astype('float')
        n = Xp.shape[0]
        C = self.n_classes
        resp = np.zeros((n, C))
        for i in range(n):
            logp_x_c = self._log_prob_x_given_c(Xp.iloc[i].to_numpy(), self.theta_) + np.log(self.pi_ + self.smooth)
            m = np.max(logp_x_c)
            p = np.exp(logp_x_c - m)
            s = p.sum()
            resp[i, :] = p / s if s > 0 else np.ones(C) / C
        return resp

# ------------- Data cleaning per Stata do-file -------------

def recode_clean(df):
    d = df.copy()
    # ensure lowercased names
    d.columns = [c.lower() for c in d.columns]

    # Binary knowledge list
    binary = ["hotcore","radioact","boyorgrl","lasers","electron","viruses","earthsun","condrift","bigbang","evolved","expdesgn","odds1","odds2"]

    # Create *_clean from binary with initial missing mapping like Stata: recode i 1=. 5=.
    for v in binary:
        if v in d.columns:
            vc = d[v].copy()
            vc = vc.astype('float')
            vc = vc.where(~vc.isin([1,5]), np.nan)
            d[v+"_clean"] = vc

    # Mapping correct vs wrong
    true_vars = ["hotcore","boyorgrl","electron","earthsun","condrift","bigbang","evolved","odds2"]
    false_vars = ["radioact","lasers","viruses","expdesgn","odds1"]

    for v in true_vars:
        c = v+"_clean"
        if c in d.columns:
            # recode 2=1 3=0 4=0
            d.loc[d[c]==2, c] = 1
            d.loc[d[c].isin([3,4]), c] = 0
    for v in false_vars:
        c = v+"_clean"
        if c in d.columns:
            # recode 3=1 2=0 4=0
            d.loc[d[c]==3, c] = 1
            d.loc[d[c].isin([2,4]), c] = 0

    # scistudy: recode 1=. 5=. 6=.
    if 'scistudy' in d.columns:
        c = 'scistudy_clean'
        d[c] = d['scistudy'].astype('float')
        d.loc[d[c].isin([1,5,6]), c] = np.nan

    # Scales: nextgen, toofast, advfront: recode 1=. 2->1 3->2 4->3 5->4 6=. 7=.
    for v in ['nextgen','toofast','advfront']:
        if v in d.columns:
            c = v+"_clean"
            d[c] = d[v].astype('float')
            d.loc[d[c]==1, c] = np.nan
            d.loc[d[c]==2, c] = 1
            d.loc[d[c]==3, c] = 2
            d.loc[d[c]==4, c] = 3
            d.loc[d[c]==5, c] = 4
            d.loc[d[c].isin([6,7]), c] = np.nan

    # scibnfts: recode 1=. 2->4 3->2 4->0 5=. 6=.
    if 'scibnfts' in d.columns:
        c = 'scibnfts_clean'
        d[c] = d['scibnfts'].astype('float')
        d.loc[d[c]==1, c] = np.nan
        d.loc[d[c]==2, c] = 4
        d.loc[d[c]==3, c] = 2
        d.loc[d[c]==4, c] = 0
        d.loc[d[c].isin([5,6]), c] = np.nan

    # Reverse toofast_clean: 4->1 3->2 2->3 1->4
    if 'toofast_clean' in d.columns:
        m = {4:1, 3:2, 2:3, 1:4}
        d['toofast_clean'] = d['toofast_clean'].map(m)

    # bible: 1=. 2->1 3->2 4->3 5=. 6=. 7=.
    if 'bible' in d.columns:
        d['bible_clean'] = d['bible'].astype('float')
        d.loc[d['bible_clean']==1, 'bible_clean'] = np.nan
        d.loc[d['bible_clean']==2, 'bible_clean'] = 1
        d.loc[d['bible_clean']==3, 'bible_clean'] = 2
        d.loc[d['bible_clean']==4, 'bible_clean'] = 3
        d.loc[d['bible_clean'].isin([5,6,7]), 'bible_clean'] = np.nan

    # reliten: 1=. 2->4 3->3 4->2 5->1 6=. 7=.
    if 'reliten' in d.columns:
        d['reliten_clean'] = d['reliten'].astype('float')
        d.loc[d['reliten_clean']==1, 'reliten_clean'] = np.nan
        d.loc[d['reliten_clean']==2, 'reliten_clean'] = 4
        d.loc[d['reliten_clean']==3, 'reliten_clean'] = 3
        d.loc[d['reliten_clean']==4, 'reliten_clean'] = 2
        d.loc[d['reliten_clean']==5, 'reliten_clean'] = 1
        d.loc[d['reliten_clean'].isin([6,7]), 'reliten_clean'] = np.nan

    return d


def listwise_delete(df, vars_clean):
    return df.dropna(subset=vars_clean)

# ------------- Labeling of classes -------------

def label_classes(profiles):
    # profiles: DataFrame with class_id index and columns: evolved_rate, knowledge_mean, bible_literalist_rate
    classes = list(profiles.index)
    # Scores
    mod_scores = profiles['knowledge_mean'] + profiles['evolved_rate'] - profiles['bible_literalist_rate']
    post_scores = profiles['knowledge_mean'] - profiles['evolved_rate'] + profiles['bible_literalist_rate']
    trad_scores = -profiles['knowledge_mean'] - profiles['evolved_rate'] + profiles['bible_literalist_rate']

    # Greedy unique assignment
    labels = {}
    # Modern
    modern = mod_scores.idxmax()
    labels[modern] = 'Modern'
    remaining = [c for c in classes if c != modern]
    # Post-secular among remaining
    postsec = post_scores.loc[remaining].idxmax()
    labels[postsec] = 'Post-secular'
    remaining = [c for c in remaining if c != postsec]
    # Traditional is the last
    if len(remaining) == 1:
        labels[remaining[0]] = 'Traditional'
    # Build mapping class_id -> label
    return labels

# ------------- Stats helpers -------------

def welch_ttest(x0, x1):
    # Returns dict with t, p, mean0, mean1, n0, n1, se_diff, ci
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)
    x0 = x0[np.isfinite(x0)]
    x1 = x1[np.isfinite(x1)]
    t, p = stats.ttest_ind(x1, x0, equal_var=False, nan_policy='omit')
    m0, m1 = x0.mean() if x0.size>0 else np.nan, x1.mean() if x1.size>0 else np.nan
    n0, n1 = x0.size, x1.size
    s0, s1 = x0.var(ddof=1), x1.var(ddof=1)
    se = np.sqrt(s0/n0 + s1/n1) if n0>1 and n1>1 else np.nan
    # 95% CI for diff (m1 - m0) using normal approx
    ci_low = (m1 - m0) - 1.96*se if np.isfinite(se) else np.nan
    ci_high = (m1 - m0) + 1.96*se if np.isfinite(se) else np.nan
    return {
        't_stat': float(t) if np.isfinite(t) else np.nan,
        'p_value': float(p) if np.isfinite(p) else np.nan,
        'mean_traditional': float(m0),
        'mean_postsecular': float(m1),
        'n_traditional': int(n0),
        'n_postsecular': int(n1),
        'se_diff': float(se) if np.isfinite(se) else np.nan,
        'ci_diff_95': [float(ci_low) if np.isfinite(ci_low) else np.nan, float(ci_high) if np.isfinite(ci_high) else np.nan]
    }


def proportion_ztest(x0, x1):
    # x0, x1 are 0/1 arrays
    x0 = np.asarray(x0)
    x1 = np.asarray(x1)
    x0 = x0[np.isfinite(x0)]
    x1 = x1[np.isfinite(x1)]
    n0, n1 = x0.size, x1.size
    p0 = x0.mean() if n0>0 else np.nan
    p1 = x1.mean() if n1>0 else np.nan
    if n0==0 or n1==0:
        return {'z_stat': np.nan, 'p_value': np.nan, 'p_traditional': float(p0), 'p_postsecular': float(p1)}
    p_pool = (x0.sum() + x1.sum()) / (n0 + n1)
    se = np.sqrt(p_pool*(1-p_pool)*(1/n0 + 1/n1))
    z = (p1 - p0) / se if se>0 else np.nan
    p = 2*(1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
    return {'z_stat': float(z) if np.isfinite(z) else np.nan, 'p_value': float(p) if np.isfinite(p) else np.nan, 'p_traditional': float(p0), 'p_postsecular': float(p1)}

# ------------- Main analysis workflow -------------

def run_subset(df, subset_name, use_year_filter=None):
    # use_year_filter: function(df)->mask or None
    d = df.copy()
    if use_year_filter is not None and 'year' in d.columns:
        d = d[use_year_filter(d)]

    # LCA indicator set as in do-file
    vars_clean = [
        'hotcore_clean','radioact_clean','boyorgrl_clean','lasers_clean','electron_clean','viruses_clean','earthsun_clean','condrift_clean','bigbang_clean','evolved_clean','expdesgn_clean','odds1_clean','odds2_clean','scistudy_clean','nextgen_clean','toofast_clean','advfront_clean','scibnfts_clean','bible_clean','reliten_clean'
    ]
    # Listwise deletion
    d2 = listwise_delete(d, [v for v in vars_clean if v in d.columns])

    # Prepare X (categorical indicators)
    X = d2[[v for v in vars_clean if v in d2.columns]].copy()

    # Convert any remaining floats to ints for categories where possible
    for col in X.columns:
        # keep NaN
        if X[col].dropna().empty:
            continue
        # round if close to int
        X[col] = X[col].apply(lambda z: np.nan if pd.isna(z) else int(z))

    weights = d2['wtss'].values if 'wtss' in d2.columns else None

    # Fit 3-class LCA with multiple starts
    lca = WeightedLCA(n_classes=3, n_starts=8, random_state=12345)
    lca.fit(X, weights=weights)
    post = lca.predict_proba(X)
    pred_class = post.argmax(axis=1)  # 0..2

    # Build class profiles
    d2 = d2.copy()
    d2['predclass'] = pred_class

    knowledge_items = [v for v in ['hotcore_clean','radioact_clean','boyorgrl_clean','lasers_clean','electron_clean','viruses_clean','earthsun_clean','condrift_clean'] if v in d2.columns]
    evolved_col = 'evolved_clean' if 'evolved_clean' in d2.columns else None

    profiles = []
    for c in range(3):
        di = d2[d2['predclass']==c]
        if len(di)==0:
            km = np.nan
            er = np.nan
            bl = np.nan
        else:
            km = float(di[knowledge_items].mean(axis=1).mean()) if knowledge_items else np.nan
            er = float(di[evolved_col].mean()) if evolved_col else np.nan
            bl = float((di['bible_clean']==1).mean()) if 'bible_clean' in di.columns else np.nan
        profiles.append({'class_id': c, 'knowledge_mean': km, 'evolved_rate': er, 'bible_literalist_rate': bl, 'n': int(len(di))})
    prof_df = pd.DataFrame(profiles).set_index('class_id')

    # Label classes
    label_map = label_classes(prof_df)
    d2['class_label'] = d2['predclass'].map(label_map)

    # Postsec vs Trad indicator
    post_id = [cid for cid,lbl in label_map.items() if lbl=='Post-secular']
    trad_id = [cid for cid,lbl in label_map.items() if lbl=='Traditional']
    post_id = post_id[0] if post_id else None
    trad_id = trad_id[0] if trad_id else None

    d2['PostsecVsTrad'] = np.where(d2['predclass']==post_id, 1, np.where(d2['predclass']==trad_id, 0, np.nan))

    # Focal tests
    if evolved_col is None:
        ttest_res = None
        ztest_res = None
    else:
        x_trad = d2.loc[d2['PostsecVsTrad']==0, evolved_col]
        x_post = d2.loc[d2['PostsecVsTrad']==1, evolved_col]
        ttest_res = welch_ttest(x_trad.values, x_post.values)
        ztest_res = proportion_ztest(x_trad.values, x_post.values)

    # Add human-readable labels into profiles
    prof_df = prof_df.copy()
    prof_df['label'] = prof_df.index.map(label_map)
    prof_df['subset'] = subset_name

    return {
        'subset': subset_name,
        'n_used': int(d2.shape[0]),
        'class_sizes': {str(lbl): int((d2['class_label']==lbl).sum()) for lbl in ['Traditional','Modern','Post-secular']},
        'profiles': prof_df.reset_index().to_dict(orient='records'),
        'welch_ttest': ttest_res,
        'two_prop_ztest': ztest_res,
        'direction': 'negative' if (ttest_res and isinstance(ttest_res.get('mean_postsecular',np.nan), float) and isinstance(ttest_res.get('mean_traditional',np.nan), float) and (ttest_res['mean_postsecular'] < ttest_res['mean_traditional'])) else 'inconclusive'
    }, prof_df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default='/app/data/GSSreplication.dta')
    parser.add_argument('--out', type=str, default='/app/data')
    args = parser.parse_args()

    data_paths = [args.data,
                  '/workspace/replication_data/GSSreplication.dta',
                  '/app/data/GSSreplication.dta',
                  '/app/data/original/6/python/replication_data/GSSreplication.dta',
                  '/workspace/replication_data/GSSreplication.csv',
                  '/app/data/GSSreplication.csv',
                  '/app/data/original/6/python/replication_data/GSSreplication.csv']

    df = None
    last_err = None
    for p in data_paths:
        if p and os.path.exists(p):
            try:
                if p.endswith('.dta'):
                    try:
                        df, meta = pyreadstat.read_dta(p)
                    except Exception:
                        df = pd.read_stata(p)
                else:
                    df = pd.read_csv(p)
                print(f"Loaded data from {p}")
                break
            except Exception as e:
                last_err = e
                continue
    if df is None:
        print("ERROR: Could not load dataset. Tried: ", data_paths)
        if last_err:
            print("Last error:", last_err)
        sys.exit(1)

    # Cleaning
    dfc = recode_clean(df)

    # Subsets: >2010, all, <=2010
    subsets = [
        ('gt2010', (lambda d: d['year']>2010) if 'year' in dfc.columns else (lambda d: pd.Series([True]*len(d), index=d.index))),
        ('all', None),
        ('le2010', (lambda d: d['year']<=2010) if 'year' in dfc.columns else (lambda d: pd.Series([True]*len(d), index=d.index)))
    ]

    results = []
    prof_out = []
    for name, filt in subsets:
        res, prof = run_subset(dfc, name, use_year_filter=filt)
        results.append(res)
        prof_out.append(prof)

    # Save outputs
    os.makedirs(args.out, exist_ok=True)
    out_json = os.path.join(args.out, 'obrien_lca_replication_results.json')
    with open(out_json, 'w') as f:
        json.dump({'subsets': results}, f, indent=2)

    class_profiles = pd.concat(prof_out, axis=0, ignore_index=True)
    out_csv = os.path.join(args.out, 'obrien_lca_class_profiles.csv')
    class_profiles.to_csv(out_csv, index=False)

    print(f"Wrote results to {out_json} and class profiles to {out_csv}")

if __name__ == '__main__':
    main()
