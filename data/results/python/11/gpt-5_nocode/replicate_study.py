#!/usr/bin/env python3
import os
import sys
import json
import re
import warnings
from typing import List, Dict, Any

import numpy as np
import pandas as pd
import pandas.api.types as pat
import statsmodels.api as sm
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

try:
    import pyreadr
except ImportError as e:
    pyreadr = None
    warnings.warn("pyreadr not installed. Please ensure pyreadr is available to read .rds files.")

try:
    from scipy import stats as sps
except Exception as e:
    sps = None

OUTPUT_DIR = "/app/data"
DEFAULT_INPUT = os.path.join(OUTPUT_DIR, "Final replication dataset.rds")


def save_json(obj: Any, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def write_text(text: str, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        f.write(text)


def load_rds(path: str) -> pd.DataFrame:
    if pyreadr is None:
        raise ImportError("pyreadr is required to load RDS files. Install pyreadr.")
    res = pyreadr.read_r(path)
    # pyreadr returns a dict-like; take the first object
    if len(res.keys()) == 0:
        raise ValueError("RDS file contains no objects.")
    key = list(res.keys())[0]
    obj = res[key]
    if isinstance(obj, pd.DataFrame):
        return obj
    else:
        # try to coerce to DataFrame if possible
        try:
            return pd.DataFrame(obj)
        except Exception as e:
            raise ValueError(f"Unsupported RDS content type: {type(obj)}")


def normalize_cols(df: pd.DataFrame) -> List[str]:
    cols = []
    for c in df.columns:
        if isinstance(c, str):
            cols.append(c)
        else:
            cols.append(str(c))
    df.columns = cols
    return cols


def col_matches(col: str, patterns: List[str]) -> bool:
    col_l = col.lower()
    return any(p in col_l for p in patterns)


def find_single_column(df: pd.DataFrame, patterns: List[str]) -> str:
    candidates = [c for c in df.columns if col_matches(c, patterns)]
    # Prefer exact-ish matches by sorting shorter names first
    candidates = sorted(candidates, key=lambda x: len(x))
    return candidates[0] if candidates else ''


def find_multiple_columns(df: pd.DataFrame, patterns: List[str], max_n: int = None) -> List[str]:
    candidates = [c for c in df.columns if col_matches(c, patterns)]
    # remove duplicates and sort for stability
    uniq = sorted(list(dict.fromkeys(candidates)))
    if max_n is not None:
        return uniq[:max_n]
    return uniq


def encode_categorical(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    cat_cols = []
    for c in columns:
        if c and c in df.columns:
            if df[c].dtype == 'O' or str(df[c].dtype).startswith('category'):
                cat_cols.append(c)
            elif df[c].nunique() <= 10 and not np.issubdtype(df[c].dtype, np.number):
                cat_cols.append(c)
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)
    return df


def to_numeric_safe(series: pd.Series) -> pd.Series:
    if np.issubdtype(series.dtype, np.number):
        return series
    try:
        return pd.to_numeric(series, errors='coerce')
    except Exception:
        return series


def standardize(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors='coerce')
    mu = s.mean()
    sd = s.std(ddof=1)
    if sd is None or sd == 0 or np.isnan(sd):
        return s * 0.0
    return (s - mu) / sd


def pool_mi(estimates: List[float], variances: List[float]) -> Dict[str, Any]:
    m = len(estimates)
    Q_bar = float(np.mean(estimates))
    U_bar = float(np.mean(variances))
    B = float(np.var(estimates, ddof=1)) if m > 1 else 0.0
    T = U_bar + (1 + 1/m) * B
    se = float(np.sqrt(T)) if T >= 0 else np.nan
    # degrees of freedom (old Rubin)
    if B == 0:
        df = 10**6  # effectively infinite
    else:
        df = (m - 1) * (1 + U_bar / ((1 + 1/m) * B)) ** 2
    t_stat = Q_bar / se if se and se > 0 else np.nan
    if sps is not None and not np.isnan(t_stat):
        try:
            pval = 2 * (1 - sps.t.cdf(abs(t_stat), df))
        except Exception:
            # fallback normal
            pval = 2 * (1 - sps.norm.cdf(abs(t_stat)))
    else:
        pval = np.nan
    return {
        'm': m,
        'Q_bar': Q_bar,
        'U_bar': U_bar,
        'B': B,
        'T': T,
        'se': se,
        'df': df,
        't': t_stat,
        'p_value': pval
    }


def main():
    in_path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_INPUT
    if not os.path.exists(in_path):
        raise FileNotFoundError(f"Input data not found at {in_path}. Place the RDS file at this path.")

    df = load_rds(in_path)
    normalize_cols(df)

    # Save column names for transparency
    write_text("\n".join(df.columns), os.path.join(OUTPUT_DIR, "dataset_columns.txt"))

    # Heuristic variable mapping    # Heuristic variable mapping
    mapping = {
        'outcome_english': find_single_column(df, patterns=[
            'english', 'cloze', 'eng_wle', 'wle_english', 'engscore', 'eng_wle_score'
        ]),
        'bilingual': find_single_column(df, patterns=[
            'bilingual', 'minority', 'home_language', 'language_home', 'lang_home', 'biling'
        ]),
        'age': find_single_column(df, patterns=['age', 'age_month', 'age_year']),
        'gender': find_single_column(df, patterns=['gender', 'sex', 'female', 'male', 'geschlecht']),
        'hisei': find_single_column(df, patterns=['hisei', 'isei']),
        'parent_education': find_single_column(df, patterns=['parent_edu', 'parental_edu', 'education_parent', 'parent_educ', 'eltern_bildung']),
        'books_home': find_single_column(df, patterns=['books', 'books_home', 'bookshome', 'buecher']),
        'german_reading': find_single_column(df, patterns=['german_read', 'german_reading', 'german_wle', 'german']),
        'class_cluster': find_single_column(df, patterns=['class', 'classid', 'klasse', 'classroom', 'clazz', 'cluster'])
    }

    # Dataset-specific fallbacks (e.g., international foreign language assessment schema)
    # Outcome: average plausible values for reading if WLE English not found
    if not mapping['outcome_english']:
        pv_read_cols = [c for c in df.columns if re.fullmatch(r'PV[1-5]_READ', c.strip(), flags=re.IGNORECASE)]
        if pv_read_cols:
            # Compute row-wise mean of available PVs
            df['english_read_mean'] = pd.to_numeric(df[pv_read_cols[0]], errors='coerce')
            if len(pv_read_cols) > 1:
                df['english_read_mean'] = pd.concat([pd.to_numeric(df[c], errors='coerce') for c in pv_read_cols], axis=1).mean(axis=1)
            mapping['outcome_english'] = 'english_read_mean'
        else:
            # alternative: any LISTening or WRITing PVs
            pv_list_cols = [c for c in df.columns if re.fullmatch(r'PV[1-5]_LIST', c.strip(), flags=re.IGNORECASE)]
            pv_writ_cols = [c for c in df.columns if re.fullmatch(r'PV[1-5]_WRIT_C', c.strip(), flags=re.IGNORECASE)]
            candidate = pv_read_cols or pv_list_cols or pv_writ_cols
            if candidate:
                df['english_pv_mean'] = pd.concat([pd.to_numeric(df[c], errors='coerce') for c in candidate], axis=1).mean(axis=1)
                mapping['outcome_english'] = 'english_pv_mean'

    # Bilingual: known binary codes fallback
    if not mapping['bilingual']:
        for cand in ['I03_ST_A_S26B', 'I03_ST_A_S27B']:
            if cand in df.columns:
                mapping['bilingual'] = cand
                break

    # Other covariates fallbacks for this dataset
    if not mapping['age'] and 'I08_ST_A_S02A' in df.columns:
        mapping['age'] = 'I08_ST_A_S02A'
    if not mapping['gender'] and 'SQt01i01' in df.columns:
        mapping['gender'] = 'SQt01i01'
    if not mapping['hisei'] and 'HISEI' in df.columns:
        mapping['hisei'] = 'HISEI'
    if not mapping['parent_education'] and 'PARED' in df.columns:
        mapping['parent_education'] = 'PARED'
    if not mapping['books_home'] and 'SQt21i01' in df.columns:
        mapping['books_home'] = 'SQt21i01'

    # Cognitive ability may have multiple indicators; collect broadly
    cognitive_cols = find_multiple_columns(df, patterns=['cft', 'cog', 'analog', 'iq', 'intelligen'])

    detected = {**mapping, 'cognitive_cols': cognitive_cols}
    save_json(detected, os.path.join(OUTPUT_DIR, 'replication_columns_detected.json'))

    required = ['outcome_english', 'bilingual']
    missing_required = [k for k in required if not mapping.get(k)]
    if missing_required:
        # Generate a data probe to help identify proper mappings
        probe = {}
        try:
            for col in df.columns:
                s = df[col]
                try:
                    sample_vals = pd.unique(s.dropna().astype(str))[:10].tolist()
                except Exception:
                    sample_vals = []
                try:
                    vc_dict = s.value_counts(dropna=True).head(10).to_dict()
                    vc_dict = {str(k): int(v) for k, v in vc_dict.items()}
                except Exception:
                    vc_dict = {}
                try:
                    nunique = int(s.nunique(dropna=True))
                except Exception:
                    nunique = 0
                probe[col] = {
                    'dtype': str(s.dtype),
                    'n_unique': nunique,
                    'sample_values': sample_vals,
                    'top_values': vc_dict
                }
        except Exception:
            pass
        try:
            save_json(probe, os.path.join(OUTPUT_DIR, 'data_probe.json'))
            write_text("Missing required columns: " + ", ".join(missing_required) + ". See dataset_columns.txt and data_probe.json for guidance.", os.path.join(OUTPUT_DIR, 'replication_error.txt'))
        except Exception:
            pass
        raise RuntimeError(f"Could not detect required columns: {missing_required}. See dataset_columns.txt and data_probe.json for guidance.")

    # Prepare dataframe with selected columns
    cols_to_use = [mapping['outcome_english'], mapping['bilingual']]
    optional_controls = ['age', 'gender', 'hisei', 'parent_education', 'books_home']
    for k in optional_controls:
        if mapping.get(k):
            cols_to_use.append(mapping[k])
    # add cognitive cols
    cols_to_use.extend([c for c in cognitive_cols if c])

    # cluster id
    cluster_col = mapping.get('class_cluster') if mapping.get('class_cluster') else None
    if cluster_col and cluster_col not in cols_to_use:
        cols_to_use.append(cluster_col)

    data = df[cols_to_use].copy()

    # Ensure bilingual is numeric binary 0/1
    bil_col = mapping['bilingual']
    if not pat.is_numeric_dtype(data[bil_col]):
        # try to map typical categories
        bil_lower = data[bil_col].astype(str).str.lower()
        if set(bil_lower.unique()) <= set(['0', '1', 'nan']):
            data[bil_col] = pd.to_numeric(bil_lower, errors='coerce')
        else:
            # Specific known coding: 0/1 already numeric as strings, or 1=Yes bilingual
            if set(bil_lower.unique()) <= set(['0', '1', 'nan', 'yes', 'no']):
                data[bil_col] = np.where(bil_lower.isin(['1', 'yes']), 1.0, np.where(bil_lower.isin(['0', 'no']), 0.0, np.nan))
            else:
                # map monolingual vs bilingual by keywords
                data[bil_col] = np.where(bil_lower.str.contains('bi|minority|non-german|zweisprach', regex=True), 1.0,
                                         np.where(bil_lower.str.contains('mono|german only|einsprach', regex=True), 0.0, np.nan))
    # Outcome numeric
    out_col = mapping['outcome_english']
    data[out_col] = to_numeric_safe(data[out_col])

    # Numericize controls where appropriate
    for k in ['age', 'hisei', 'parent_education']:
        c = mapping.get(k)
        if c and c in data.columns:
            data[c] = to_numeric_safe(data[c])
    # books at home: ordinal categories -> numeric ordered codes
    bcol = mapping.get('books_home')
    if bcol and bcol in data.columns and not pat.is_numeric_dtype(data[bcol]):
        order = ['0-10 books', '11-25 books', '26-100 books', '101-200 books', '201-500 books', 'More than 500 books']
        cat = pd.Categorical(data[bcol].astype(str), categories=order, ordered=True)
        codes = pd.Series(cat.codes, index=data.index)
        data[bcol] = codes.replace({-1: np.nan})

    # Encode gender (and any non-numeric controls) as dummies
    non_num_controls = []
    # gender encoding
    c = mapping.get('gender')
    if c and c in data.columns:
        if not pat.is_numeric_dtype(data[c]):
            # normalize values
            g_lower = data[c].astype(str).str.lower()
            if set(g_lower.unique()) <= set(['male', 'female', 'nan', 'm', 'f']):
                data[c] = np.where(g_lower.isin(['female', 'f']), 1.0, np.where(g_lower.isin(['male', 'm']), 0.0, np.nan))
            else:
                non_num_controls.append(c)
    if non_num_controls:
        data = encode_categorical(data, non_num_controls)

    # Add cognitive columns (numeric)
    for c in cognitive_cols:
        if c in data.columns:
            data[c] = to_numeric_safe(data[c])

    # Separate cluster ids
    clusters = None
    if cluster_col and cluster_col in data.columns:
        clusters = data[cluster_col]
        # keep cluster as object; ensure not imputed
        data = data.drop(columns=[cluster_col])

    # Define predictors list
    predictors = [bil_col]
    control_cols = []
    for k in ['age', 'gender', 'hisei', 'parent_education', 'books_home']:
        c = mapping.get(k)
        # gender may have expanded into dummies; handle pattern
        if k == 'gender' and c and c not in data.columns:
            # add any dummies that start with c + '_'
            gender_dummies = [col for col in data.columns if col.startswith(c + '_')]
            control_cols.extend(gender_dummies)
            continue
        if c and c in data.columns:
            control_cols.append(c)
    # cognitive controls
    control_cols.extend([c for c in cognitive_cols if c in data.columns])

    X_cols = list(dict.fromkeys(predictors + control_cols))
    # Ensure X_cols do not include the outcome column
    X_cols = [c for c in X_cols if c != out_col]

    # Imputation setup
    m = 5
    rng_seed = 12345

    # Build imputation frame limited to outcome + predictors
    impute_cols = [out_col] + X_cols
    impute_df = data[impute_cols].copy()

    # Drop predictors that are entirely missing to avoid sklearn dropping them implicitly
    all_nan_predictors = [c for c in X_cols if impute_df[c].isna().all()]
    if all_nan_predictors:
        X_cols = [c for c in X_cols if c not in all_nan_predictors]
        impute_cols = [out_col] + X_cols
        impute_df = impute_df[impute_cols]
        # Log which were dropped
        try:
            write_text("Dropped all-NaN predictors: " + ", ".join(all_nan_predictors), os.path.join(OUTPUT_DIR, 'imputation_dropped_predictors.txt'))
        except Exception:
            pass

    # Build imputers and perform m imputations
    imputations = []
    imputer = IterativeImputer(random_state=rng_seed, sample_posterior=True, max_iter=25, initial_strategy='median')
    # Fit once to get model, then sample multiple imputations by varying random_state
    imputer.fit(impute_df)
    for i in range(m):
        imputer.random_state = rng_seed + i
        imp_array = imputer.transform(impute_df)
        imp_df = pd.DataFrame(imp_array, columns=impute_df.columns, index=impute_df.index)
        imputations.append(imp_df)

    # Fit models and collect estimates
    param_name = bil_col
    estimates = []
    variances = []
    full_results = []

    for i, dfi in enumerate(imputations):
        # Add constant
        X = dfi[X_cols].copy()
        X = sm.add_constant(X, has_constant='add')
        y = dfi[out_col]
        # clusters (use original clusters; for any missing, drop corresponding rows)
        if clusters is not None:
            # Align indices
            clust_i = clusters.loc[dfi.index]
            # Drop any rows with missing cluster ids
            keep = clust_i.notna() & y.notna()
            X_i = X.loc[keep]
            y_i = y.loc[keep]
            clust_i = clust_i.loc[keep]
            model = sm.OLS(y_i, X_i)
            res = model.fit(cov_type='cluster', cov_kwds={'groups': clust_i})
        else:
            keep = y.notna()
            X_i = X.loc[keep]
            y_i = y.loc[keep]
            model = sm.OLS(y_i, X_i)
            res = model.fit(cov_type='HC1')
        full_results.append({
            'params': res.params.to_dict(),
            'bse': res.bse.to_dict(),
            'cov': res.cov_params().values.tolist(),
            'cov_index': res.cov_params().index.tolist(),
            'nobs': float(res.nobs)
        })
        # Extract bilingual term
        if param_name not in res.params.index:
            raise RuntimeError(f"Bilingual variable '{param_name}' not in regression parameters for imputation {i}.")
        beta = float(res.params[param_name])
        var = float(res.cov_params().loc[param_name, param_name])
        estimates.append(beta)
        variances.append(var)

    pooled = pool_mi(estimates, variances)

    results = {
        'model': 'OLS with background controls (cluster-robust if cluster id available)',
        'outcome': out_col,
        'predictors': X_cols,
        'cluster_col': cluster_col,
        'bilingual_param': param_name,
        'mi_pooled': pooled,
        'per_imputation': full_results
    }

    save_json(results, os.path.join(OUTPUT_DIR, 'replication_results.json'))

    # Human-readable summary
    summary_lines = []
    summary_lines.append("Replication focal model: English WLE ~ Bilingual + controls")
    summary_lines.append(f"Outcome: {out_col}")
    summary_lines.append(f"Bilingual variable: {param_name}")
    summary_lines.append(f"Controls: {', '.join([c for c in X_cols if c != 'const' and c != param_name])}")
    summary_lines.append(f"Cluster column: {cluster_col if cluster_col else 'None'}")
    summary_lines.append("")
    summary_lines.append("Multiple Imputation (m=5) pooled estimate for Bilingual:")
    summary_lines.append(json.dumps(pooled, indent=2))

    write_text("\n".join(summary_lines), os.path.join(OUTPUT_DIR, 'replication_results.txt'))


if __name__ == '__main__':
    main()
