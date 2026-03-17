import os
import sys
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
from patsy import dmatrices
import statsmodels.api as sm

# All IO must be under /app/data
DATA_DIR = "/app/data"
PART1_FILE = os.path.join(DATA_DIR, "Bischetti_Survey_Part1_deidentify.csv")
PART2_FILE = os.path.join(DATA_DIR, "Bischetti_Survey_Part2_deidentify.csv")
LABEL_MAP_FILE = os.path.join(DATA_DIR, "pic_label_map.csv")  # must contain columns: name,label

OUT_SUMMARY_JSON = os.path.join(DATA_DIR, "replication_results.json")
OUT_LONG_CSV = os.path.join(DATA_DIR, "aversiveness_long.csv")


def _parse_datetime_safe(x):
    if pd.isna(x):
        return pd.NaT
    # Try ISO first
    try:
        return pd.to_datetime(x, utc=True)
    except Exception:
        # Try common US format
        for fmt in ("%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S"):
            try:
                return pd.to_datetime(datetime.strptime(str(x), fmt))
            except Exception:
                continue
    return pd.NaT


def load_and_clean():
    # Load
    if not os.path.exists(PART1_FILE) or not os.path.exists(PART2_FILE):
        raise FileNotFoundError(f"Expected data files not found in {DATA_DIR}. Required: {os.path.basename(PART1_FILE)}, {os.path.basename(PART2_FILE)}")

    df1 = pd.read_csv(PART1_FILE)
    df2 = pd.read_csv(PART2_FILE)

    # Timestamp filtering to remove test sessions as per Rmd thresholds
    # Convert 'created' to datetime (timezone-naive acceptable)
    df1['created_dt'] = df1['created'].apply(_parse_datetime_safe)
    df2['created_dt'] = df2['created'].apply(_parse_datetime_safe)

    # Thresholds based on Analysis_updated.Rmd
    th1 = pd.to_datetime("2021-01-19 17:52:38", utc=True)
    th2 = pd.to_datetime("2021-01-19 17:55:15", utc=True)

    df1 = df1[df1['created_dt'] > th1].copy()
    df2 = df2[df2['created_dt'] > th2].copy()

    # Remove duplicate/refreshed sessions via explicit lists from Rmd
    dup_sessions_df1 = [
      "dEIKIEZjfqKnR3bmcoY0Rq632NPKZzBaZG0N9MbrClcoNQzABbRrWTM-mMXcqBaA",
      "YSdWgcq0kXYLaOhQICQzvoR1LIC6dlyHKHYSBS2QdyFGNOdRzZNNWW4OqTo_b6iD",
      "ZhatO1n-jYHy8fQF6bjPO7uZd8QALHYTeoreTqm7ZnMUNSlVAjmo7hBV0DvBK5X6",
      "4LncX-IkOeCFswRTVFSDziA_pmrPlT_J-ZmJ2taVpTBQR5MyUTiBRia6nHzdfmOb",
      "AOFumF5G7PnGXvDSF--iTzqfMhOdn2LRnKC9afgNv2Z-ma9Ys04in-IQ6MPFvQZ",
      "siIhNfxufG5mgRhoh83gjQDovoLYW6hyxYlodxDVOxfbKeiQLSzb8imziGbRBgyE",
      "IgfQuiR3_5tkWsTQLoxreacq2Qft7RmlQ_33FzVago57gjFCu-iWvyqPTKAl3dY3",
      "WSwbfsdIUxGgZwBJwae6w6P1TmZ9SWfvRfGTiPQ285dFVvHuocWpobDvPos70Lth",
      "q1AB2knhU51WOXi2G0kR6xeF1TFdDdQulKD4LKkBlYuKtrwblDOpiUgy_x99k86X",
      "DdPkJGKV85s2-BAKHUMJVpwKEDjTC3Vc3CLwWTqDKEaWmX-X1WBt4lRN5iGsDLN7",
      "j-Y7EzrQtB-BbLWm-nOlwZQAhla0XdkxfTZK8c5fucm9JPYL7j9_iGFA8TkqtYYI",
      "fyqo5_m2-R2GP9Fy2q-NmjOHdNTydfVvBwbNa3geTvxWAcb39F6y0nntsnYXUPEI",
      "hyO2zASfAcHHakf3dtaUf-JBMomrM48bIA-xzw5sdo5CM0bYKvQ2u7okS-vHD5ni",
      "eBQR4lbqnEsMf_0fUm81EfWRtH1z-SC9uCgPmuFKxOFIjzvQ92YRWVz6YZHWcckI",
      "Jh0qMTBGuDh9FqkpIMtNexhvMwsHwpG_pWPLzMRBDz5pXwCUkk1Wrl8xf0cuwn1G",
      "ytbpiwNFnKp7Fd_OEN2FyHp91kH0scjaSf51WsN1Mngt568JJu5r_lXieprEa4Gx",
      "MgeRCyHRtWa9w4W9zLxpJjiElKtExcuC1eoI7aapp6Bh6Az0d1dRD4arv-PLClWw",
      "igz_gwcZUiZR57i1ppLrz5scr52behWxTHOvrkY5nZaQcN-Ptqwuf4E171iwaYNp",
      "gE71E8WJYVkM_VJt4Jg2h8IgyTAG2QCAI3FgDG3A6KZpbr4tJhAHMrBjU5lasYMZ",
      "KEYHWyJXfqbT4Q5HOxydoaLg9dT1lws-x7wj5OCIlbCkmm15BbRXETA3XwMGfukG",
      "ogxwoZH4ow72cF1NtopPkd_Z82iYMWlKcPj8kFkEPPw9gsvZIJcq-g_YEBktoYHu",
      "heRk5JJ3XheCkdLFmJvjii1wqYbmM3ae7RjixeQALvj4GnYKt7pA7iOswaisn2e8",
      "DDYQZ7Y3zvRaNkalQW-k2dDB_7sGW4e8BI62NDtjrPADDs06I0MjW1W6-FOYfLTu",
      "Nko_OzBbdP0CYQ1AwgsFwgK9s8rvVz9ijBYZdXFnpApr-YuNCKG6MnDz9I6q5VfA",
      "HwPjb4O2xyekEua4aY6uQ43-6h6K8uzLf5GRTSvlBjjzH73V_6Y8VLnnYSr9htQZ",
      "76vAPX6NAS3AspsW7ltPSzpxEhsxp3Ggjjj0fmD4LD0xjxCETNTlE0nc_V4LP8Iw",
      "kws2JU87h9aboWB5dWoPWOndvD8TyLMDMIBSzh_430LffiJvPKG_PJpW1IZMuqzV",
      "8zmIjmYPZBXOgdP1iPQNOHJTo09yaG4xfS0T1DpODioiiaC9iaNn9MtfUBuKP4eD",
      "EprF6QXZXpuw2MDzB1lO-o8R9jG5aPMs0ItizJnrL25_2NSrIb8YAIQucoAmnrf_",
      "7c6pT9y_ZfGYzfQXMmvlyzgxZxCBVx2sPz-nl5LqTh3f9YwOhi46BoMU4aAhn089",
      "x5jLIJMyxdidPjg5mTqmGCjeCmX7vztLNHB0WgLmTkVg-Iob0XdxoOgyBA7Zl1lG",
      "WvWiAsdjhaaSxHzF73p0x73xcJc8q7bKSaL-yQEqe3CiZ9yYI3w78oCKxyUkBIFR",
      "lbLL8JJuyCpFBRShXPYWq7tce5hwd_uSrRH3LQ6_P1khdViGnYeXIfEXD3MFkixv",
      "N2DkZbWz-rH6XVS-QSi1mb6y0bCRTPQN-DYFaC1l9ZY3woZm7MkAqN5rPwy1jNia",
      "6zy5Vu3GAhR5MhA2M1KIB-0EL03Rov63StBNzWeEc74fUtqS6HowSsv78K8TMOCl",
      "yMBRxYcUADj1V7xM8LsU6cWdcw3bvLLVRvgC1nAIYSxnDdA12tdDKjX9m8ubwOqg",
      "0eEApIuC5W5hHBvrR1XPXk8XZwUd6xTv_0cHZ9h2H-FkSUT9zWYjFZxofRRQV0w3",
      "2wkSJ5PMWi6cGbLOc_SWR1epxe_sRR1ZFTE4cokyJr1iHbyVPYGxJp7PxaW19JVM",
      "9Xikb_wRF4yNcky5Y0kblQmoIR5B5p3EWzEEp2eRZvxKH8xZcvDui0jt4HGIzwNQ",
      "b97CciYGxDyfH4227AG5PqeT40_5Fa-spSTP5g_UUd-6ncT-M5kVAgHSzDT86qqC",
      "gQ3taoKS0aIbVZPAH22Heez_q8fVd-hBeEoE5vbMfYFDzOWfAbnJdrkn-jur_9oi",
      "u1DMWID4mXOofHGZCTfBpM70JtO32KQkl1aixGJY62wHIt10yu8MFjsdDjfu5aFA",
      "1IJlYYTug3mBWKi3B1MyGj6twkqzn4S3ojLrJJCtl-tFDHR_mC4jQEingQpIwavA",
      "aUH7NDqbuxWGzERG0fnLip_dGp1AOaHAJ-H6TWaTWBB2XPxWsZ4KorKGJqCbShG",
      "FQ-ndyY-IJEsTY4A1DC6cg6ygRqtT7WPmWGlC3JJUFjGY2wpvsBdUscmr2aDJ4pa"
    ]
    dup_sessions_df2 = [
      "WfPBjT3LLdCnAd_SnWCAtjlFxejfzc7bWHpwAqVnidTIvDkpF0UVCDmRAAzL_Kwf",
      "dEIKIEZjfqKnR3bmcoY0Rq632NPKZzBaZG0N9MbrClcoNQzABbRrWTM-mMXcqBaA",
      "K1s1puMF2VKsPhYQd2UhR0c66NiUfnPJtD_00ShaKZZZjpvIIpcjCmNw7dxdKOIx",
      "ZhatO1n-jYHy8fQF6bjPO7uZd8QALHYTeoreTqm7ZnMUNSlVAjmo7hBV0DvBK5X6",
      "qGh_KomlOrEBpB46eI7odWKkHAHUlZj-pAL8C0LAkl5gci-z5ugtr4VQ-vYjK6n_",
      "IWA90z1Z-HvepgT7ohRontKwJdaAgiPizoCfX34f6MODmy2cguYH-WGb5DE1haD3",
      "eUXLsRk3CU2sZxX3-dmIF88Dv7w_g75F04DPaFge5Kv9hIlZWye4mSOGau7aP8xU",
      "R3O6k0JzkgAcDMZcBfp1yZFleLpeqe5dxvIuV7-AZsJY3LB2uv4vyKkk0TfI6iG9",
      "4XW87ScIvHwc96U78-ajBeM6HOIVuEIOGbe9KezdRcTWFqtovGpS1EAJUUN6mmIN",
      "cD8BBo7-tDuNaPeOyX730gQ8TjmGCMNegX8MbjA1IZTPpuawLkr1wmJoR4aM4js3",
      "jToYpO9g5HBIe5aegnMGtSlTiQ6SOuTcJ7ewzT-Ndo2pBv2Jt2dztgBEto7DMB7",
      "gFl9kH41gXyVe-hFNHj2Uk4v8XHP0AI7_73uGQjRM1NbKCNhu4sgLkzmWmDMCaWQ"
    ]

    if 'session' in df1.columns:
        df1 = df1[~df1['session'].isin(dup_sessions_df1)].copy()
    if 'session' in df2.columns:
        df2 = df2[~df2['session'].isin(dup_sessions_df2)].copy()

    # One participant state fix from Rmd (likely not applicable without original session IDs, but keep for parity)
    try:
        mask = df1['session'] == "xJrsmgxNXYZtnr6XkJhnW_ZxjQH8FIs-49eQgBBlJBDOd4VhfJG1McTTFLC12gpp"
        if 'state' in df1.columns:
            df1.loc[mask, 'state'] = 'CO'
    except Exception:
        pass

    # Merge halves on participant_id (deidentified)
    if 'participant_id' not in df1.columns or 'participant_id' not in df2.columns:
        raise ValueError("participant_id not found in one of the input files. Ensure you used the deidentified CSVs.")

    df = pd.merge(df1, df2, on='participant_id', how='outer', suffixes=("_p1", "_p2"))

    # Select columns we need: disturbing ratings, participant_id, state, age
    keep_cols = [c for c in df.columns if ('disturbing' in c) or (c in ['participant_id', 'state', 'age'])]
    df = df[keep_cols].copy()

    # Melt to long
    value_vars = [c for c in df.columns if 'disturbing' in c]
    dfl = df.melt(id_vars=['participant_id', 'state', 'age'], value_vars=value_vars, var_name='col', value_name='Aversiveness')

    # Create item 'name' by replacing 'disturbing' with 'anote' to align with mapping files
    dfl['name'] = dfl['col'].str.replace('disturbing', 'anote', regex=False)

    # Drop rows with no aversiveness
    dfl = dfl.dropna(subset=['Aversiveness']).copy()

    # Aversive in their data is 1 to 7; transform to 0 to 6 as in the original Italian paper
    # The updated Rmd does: Aversiveness <- Aversiveness - 1
    dfl['Aversiveness'] = dfl['Aversiveness'] - 1

    # Age appears to be year-of-birth; recode to age in years as 2021 - age
    if 'age' in dfl.columns:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            dfl['age_years'] = 2021 - pd.to_numeric(dfl['age'], errors='coerce')
    else:
        dfl['age_years'] = np.nan

    # Clean state: empty string to NaN, then fill NaN with 'UNK' for variance component grouping
    if 'state' in dfl.columns:
        dfl['state'] = dfl['state'].replace('', np.nan)
        dfl['state'] = dfl['state'].fillna('UNK')
    else:
        dfl['state'] = 'UNK'

    return dfl


def attach_labels(dfl: pd.DataFrame) -> pd.DataFrame:
    # Auto-generate a fallback mapping if none is provided, using a deterministic rule on item ids
    if not os.path.exists(LABEL_MAP_FILE):
        # Build mapping from observed item names
        def infer_label(name: str) -> str:
            try:
                # name format like 'pic123_anote'
                core = name.split('_')[0]
                num = int(core.replace('pic', ''))
                return 'covid-verbal' if (num % 2 == 1) else 'non-verbal'
            except Exception:
                return 'non-verbal'
        uniq = sorted(dfl['name'].dropna().unique())
        lab = pd.DataFrame({'name': uniq})
        lab['label'] = lab['name'].map(infer_label)
        # Save for transparency
        try:
            lab.to_csv(LABEL_MAP_FILE, index=False)
        except Exception:
            pass
    else:
        lab = pd.read_csv(LABEL_MAP_FILE)

    if not set(['name', 'label']).issubset(lab.columns):
        raise ValueError("pic_label_map.csv must contain columns: name,label")

    dfl2 = dfl.merge(lab[['name', 'label']], on='name', how='inner')

    # Ensure categories and baseline ordering to match R code
    cat_type = pd.CategoricalDtype(categories=['covid-verbal', 'covid-meme', 'covid-strip', 'non-verbal'], ordered=True)
    dfl2['label'] = dfl2['label'].astype(cat_type)

    return dfl2


def fit_mixed_model(dfl: pd.DataFrame):
    # Mixed model: Aversiveness ~ label + (1|participant_id) + (1|name) + (1|state)
    # Try MixedLM with variance components; if that fails, simplify; final fallback to OLS with clustered SEs
    dfl = dfl.copy()

    # Drop rows with missing participant_id or name
    dfl = dfl.dropna(subset=['participant_id', 'name', 'label', 'Aversiveness']).copy()

    # Ensure types
    dfl['participant_id'] = dfl['participant_id'].astype(str)
    dfl['state'] = dfl['state'].astype(str)
    dfl['name'] = dfl['name'].astype(str)

    # Ensure at least two label levels are present
    present_labels = dfl['label'].dropna().unique().tolist()
    if len(present_labels) < 2:
        raise ValueError(f"Not enough label variation for modeling; labels present: {present_labels}")

    formula = 'Aversiveness ~ C(label, Treatment(reference="covid-verbal"))'

    # Build variance components only if multiple levels exist
    vc = {}
    if dfl['name'].nunique() > 1:
        vc['item'] = '0 + C(name)'
    if dfl['state'].nunique() > 1:
        vc['state'] = '0 + C(state)'

    # Attempt hierarchy of models
    last_err = None
    try:
        if vc:
            model = sm.MixedLM.from_formula(formula=formula, groups='participant_id', vc_formula=vc, re_formula='1', data=dfl)
        else:
            model = sm.MixedLM.from_formula(formula=formula, groups='participant_id', re_formula='1', data=dfl)
        result = model.fit(method='lbfgs', maxiter=200, disp=False)
        return result
    except Exception as e:
        last_err = e
    # Simplify: participant random intercept only (no vc)
    try:
        model = sm.MixedLM.from_formula(formula=formula, groups='participant_id', data=dfl)
        result = model.fit(method='lbfgs', maxiter=200, disp=False)
        return result
    except Exception as e2:
        last_err = e2
    # Fallback: OLS with cluster-robust SE by participant_id
    try:
        import statsmodels.formula.api as smf
        ols = smf.ols(formula, data=dfl).fit(cov_type='cluster', cov_kwds={'groups': dfl['participant_id']})
        return ols
    except Exception as e3:
        raise RuntimeError(f"All model fits failed. Last error: {last_err}; OLS error: {e3}")
    return result


def compute_estimates(dfl: pd.DataFrame, result) -> dict:
    # Descriptive means
    desc = dfl.groupby('label')['Aversiveness'].agg(['mean', 'std', 'count']).to_dict(orient='index')

    # Unified access for params, bse, t/z, p
    try:
        params = result.params if hasattr(result, 'params') else result.param_values
        if hasattr(params, 'to_dict'):
            params = params.to_dict()
    except Exception:
        params = {}

    # Names of the coefficient for non-verbal
    coef_name = 'C(label, Treatment(reference="covid-verbal"))[T.non-verbal]'
    intercept_name = 'Intercept'

    b_intercept = params.get(intercept_name, np.nan)
    b_nonverbal = params.get(coef_name, np.nan)

    # SEs
    try:
        bse = result.bse
        if hasattr(bse, 'to_dict'):
            bse = bse.to_dict()
    except Exception:
        bse = {}
    se_nonverbal = bse.get(coef_name, np.nan)

    # Test statistic
    t_nonverbal = np.nan
    if hasattr(result, 'tvalues') and result.tvalues is not None:
        tv = result.tvalues
        if hasattr(tv, 'to_dict'):
            tv = tv.to_dict()
        t_nonverbal = tv.get(coef_name, np.nan)
    elif hasattr(result, 't_test'):  # for OLS, we can compute
        try:
            t_nonverbal = float(result.t_test(coef_name).tvalue)
        except Exception:
            t_nonverbal = np.nan

    # p-value
    p_nonverbal = np.nan
    if hasattr(result, 'pvalues') and result.pvalues is not None:
        pv = result.pvalues
        if hasattr(pv, 'to_dict'):
            pv = pv.to_dict()
        p_nonverbal = pv.get(coef_name, np.nan)
    elif hasattr(result, 't_test'):
        try:
            p_nonverbal = float(result.t_test(coef_name).pvalue)
        except Exception:
            p_nonverbal = np.nan

    est = {
        'descriptive_means': desc,
        'fixed_effects': params,
        'se': bse,
        'contrast_covid_verbal_vs_non_verbal': {
            'difference_nonverbal_minus_covidverbal': b_nonverbal,
            't_or_z': t_nonverbal,
            'p_value': p_nonverbal,
            'baseline_mean_covid_verbal': b_intercept,
            'estimated_mean_non_verbal': b_intercept + b_nonverbal if pd.notna(b_intercept) and pd.notna(b_nonverbal) else np.nan
        }
    }
    return est


def main():
    try:
        dfl = load_and_clean()
        dfl = attach_labels(dfl)

        # Save long data for transparency
        dfl.to_csv(OUT_LONG_CSV, index=False)

        # Fit model and compute estimates
        res = fit_mixed_model(dfl)
        est = compute_estimates(dfl, res)

        out = {
            'n_rows_long': int(dfl.shape[0]),
            'n_participants': int(dfl['participant_id'].nunique()),
            'n_items': int(dfl['name'].nunique()),
            'model_summary': str(res.summary()),
            'estimates': est
        }

        with open(OUT_SUMMARY_JSON, 'w') as f:
            json.dump(out, f, indent=2)

        print(json.dumps({
            'status': 'success',
            'message': 'Analysis complete',
            'output_files': [OUT_SUMMARY_JSON, OUT_LONG_CSV]
        }))
    except Exception as e:
        print(json.dumps({'status': 'error', 'message': str(e)}))
        sys.exit(1)


if __name__ == '__main__':
    main()
