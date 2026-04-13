import os
import json
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.api as sm

ARTIFACT_DIR = "/app/artifacts"
DATA_DIR = next((d for d in ["/app/data/replication_data", "/workspace/replication_data", os.path.join(os.getcwd(), "replication_data")] if os.path.isdir(d)), "/workspace/replication_data")

os.makedirs(ARTIFACT_DIR, exist_ok=True)


def to_year_from_stata_date(series):
    # Try to parse Stata daily date (days since 1960-01-01)
    try:
        if np.issubdtype(series.dropna().values[:1].dtype, np.number):
            dt = pd.to_datetime(series, unit="D", origin="1960-01-01", errors="coerce")
        else:
            dt = pd.to_datetime(series, errors="coerce")
        return dt.dt.year
    except Exception:
        # Fallback: try generic parsing
        dt = pd.to_datetime(series, errors="coerce")
        return dt.dt.year


def first_present_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def main():
    log_lines = []
    try:
        # 1) Load datasets and drop duplicates
        cpds_path = os.path.join(DATA_DIR, "CPDS_final.dta")
        cmp_path = os.path.join(DATA_DIR, "CMP_final.dta")

        if not os.path.exists(cpds_path) or not os.path.exists(cmp_path):
            raise FileNotFoundError(f"Required data files not found in {DATA_DIR}.")

        cpds = pd.read_stata(cpds_path, convert_categoricals=False)
        cmp = pd.read_stata(cmp_path, convert_categoricals=False)

        cpds = cpds.drop_duplicates().copy()
        cmp = cmp.drop_duplicates().copy()

        # 2) Harmonize key columns
        if "countryname" in cmp.columns and "country" not in cmp.columns:
            cmp = cmp.rename(columns={"countryname": "country"})

        # Create 'year' in CMP from 'edate'
        if "edate" not in cmp.columns:
            raise KeyError("CMP dataset missing 'edate' column required to derive year.")
        cmp["year"] = to_year_from_stata_date(cmp["edate"])  # numeric year
        # Create a robust numeric year for merging
        cmp["year_merge"] = pd.to_numeric(cmp["year"], errors="coerce").astype("Int64")

        # Ensure CPDS has compatible merge keys
        if "year" in cpds.columns:
            cpds["year_merge"] = pd.to_numeric(cpds["year"], errors="coerce").astype("Int64")
        elif "year_merge" in cpds.columns:
            cpds["year_merge"] = pd.to_numeric(cpds["year_merge"], errors="coerce").astype("Int64")
        else:
            # Try derive from possible date column
            if "edate" in cpds.columns:
                cpds_year = to_year_from_stata_date(cpds["edate"])  # fallback
                cpds["year_merge"] = pd.to_numeric(cpds_year, errors="coerce").astype("Int64")
            else:
                raise KeyError("CPDS dataset missing 'year' or 'edate' to merge on year.")

        if "country" not in cmp.columns:
            raise KeyError("CMP dataset missing 'country' column.")
        if "country" not in cpds.columns:
            raise KeyError("CPDS dataset missing 'country' column.")

        # 3) Merge CMP (party-level) with CPDS (country-year)
        merged = cmp.merge(cpds, how="inner", left_on=["country", "year_merge"], right_on=["country", "year_merge"], suffixes=("", "_cpds"))

        # 4) Build election identifier per country sorted by edate
        tmp = merged[["country", "edate"]].drop_duplicates().copy()
        tmp = tmp.sort_values(["country", "edate"])  # order within country by date
        tmp["election"] = tmp.groupby("country").cumcount() + 1
        merged = merged.merge(tmp, on=["country", "edate"], how="left")

        # 5) Party identifier
        party_col = first_present_column(merged, ["party", "partyname", "party_id", "partyid", "partyabbrev"])  # heuristic
        if party_col is None:
            raise KeyError("Could not find a party identifier column among ['party','partyname','party_id','partyid','partyabbrev'].")
        if party_col != "party":
            merged = merged.rename(columns={party_col: "party"})

        # 6) relative_seat = absseat / totseats
        if not set(["absseat", "totseats"]).issubset(merged.columns):
            raise KeyError("Required columns 'absseat' and/or 'totseats' missing for seat calculations.")
        merged["relative_seat"] = merged["absseat"] / merged["totseats"]

        # 7) Consecutive-election filters within party (sorted by election)
        merged = merged.sort_values(["party", "election"])  # ensure order within party
        rs = merged["relative_seat"]
        merged["rs_lead1"] = merged.groupby("party")["relative_seat"].shift(-1)
        merged["two_consec"] = (merged["relative_seat"] > 0.01) & (merged["rs_lead1"] > 0.01)
        keep_two = merged.groupby("party")["two_consec"].transform("max") == 1
        merged = merged.loc[keep_two].copy()

        # Recompute for three consecutive
        merged = merged.sort_values(["party", "election"])  # maintain order
        merged["rs_lead1"] = merged.groupby("party")["relative_seat"].shift(-1)
        merged["rs_lead2"] = merged.groupby("party")["relative_seat"].shift(-2)
        merged["three_consec"] = (merged["relative_seat"] > 0.01) & (merged["rs_lead1"] > 0.01) & (merged["rs_lead2"] > 0.01)
        keep_three = merged.groupby("party")["three_consec"].transform("max") == 1
        merged = merged.loc[keep_three].copy()

        # 8) Count parties per country-election and log-transform
        counts = merged.groupby(["country", "election"]).size().reset_index(name="count_parties")
        merged = merged.merge(counts, on=["country", "election"], how="left")
        merged["count_parties"] = np.log(merged["count_parties"].astype(float))

        # 9) PCA for economic policy dimension
        econ_vars = ["per303", "per401", "per402", "per403", "per404", "per407", "per412", "per413", "per414", "per504", "per505", "per701"]
        missing_vars = [v for v in econ_vars if v not in merged.columns]
        if missing_vars:
            raise KeyError(f"Missing economic policy variables for PCA: {missing_vars}")

        pca_data = merged[econ_vars].copy()
        # Drop rows with missing PCA inputs
        pca_mask = ~pca_data.isna().any(axis=1)
        pca_df = merged.loc[pca_mask].copy()

        scaler = StandardScaler(with_mean=True, with_std=True)
        Xs = scaler.fit_transform(pca_df[econ_vars].values)
        pca = PCA(n_components=1)
        comp_scores = pca.fit_transform(Xs)  # first component scores
        pca_df["economic_policy"] = comp_scores[:, 0]

        # merge scores back; rows without PCA inputs will lack scores and be dropped for dispersion
        merged = merged.merge(pca_df[["country", "election", "party", "economic_policy"]], on=["country", "election", "party"], how="left")

        # 10) Dispersion per country-election: max - min of economic_policy; log-transform
        disp_df = merged.dropna(subset=["economic_policy"]).copy()
        grp = disp_df.groupby(["country", "election"], as_index=False)
        dmin = grp["economic_policy"].min().rename(columns={"economic_policy": "min_econ"})
        dmax = grp["economic_policy"].max().rename(columns={"economic_policy": "max_econ"})
        disp = dmin.merge(dmax, on=["country", "election"], how="inner")
        disp["dispersion"] = disp["max_econ"] - disp["min_econ"]
        disp = disp.loc[disp["dispersion"] > 0].copy()
        disp["dispersion"] = np.log(disp["dispersion"].astype(float))

        # 11) Controls
        # single_member = (prop == 0)
        if "prop" in merged.columns:
            # Assume prop constant within country-election
            prop_ce = merged.groupby(["country", "election"], as_index=False)["prop"].first()
            prop_ce["single_member"] = (prop_ce["prop"] == 0).astype(float)
        else:
            prop_ce = merged[["country", "election"]].drop_duplicates().copy()
            prop_ce["single_member"] = np.nan

        # Year per country-election (from earlier derived numeric year)
        year_ce = merged.groupby(["country", "election"], as_index=False)["year"].first()

        # Assemble panel-level dataset (unique per country-election)
        panel = disp.merge(prop_ce[["country", "election", "single_member"]], on=["country", "election"], how="left")
        panel = panel.merge(year_ce, on=["country", "election"], how="left")

        # id_country (group code)
        panel = panel.sort_values(["country", "election"]).reset_index(drop=True)
        panel["id_country"], _ = pd.factorize(panel["country"], sort=True)

        # Lagged dependent variable by country ordered by election
        panel["lagged_dispersion"] = panel.groupby("id_country")["dispersion"].shift(1)

        # 12) Regression: dispersion ~ count_parties + single_member + lagged_dispersion, cluster(id_country)
        # Need count_parties per country-election (the log-transformed measure). Use the first party row within group.
        cp_ce = merged.groupby(["country", "election"], as_index=False)["count_parties"].first()
        panel = panel.merge(cp_ce, on=["country", "election"], how="left")

        reg_data = panel[["dispersion", "count_parties", "single_member", "lagged_dispersion", "id_country"]].copy()
        reg_data = reg_data.dropna().copy()

        if reg_data.empty or reg_data.shape[0] < 10:
            raise RuntimeError("Not enough observations after preprocessing to run regression.")

        y = reg_data["dispersion"].astype(float)
        X = reg_data[["count_parties", "single_member", "lagged_dispersion"]].astype(float)
        X = sm.add_constant(X)

        model = sm.OLS(y, X)
        try:
            res = model.fit(cov_type="cluster", cov_kwds={"groups": reg_data["id_country"]})
        except Exception as e:
            # Fallback to robust if clustering fails
            res = model.fit(cov_type="HC1")
            log_lines.append(f"Warning: Clustered SE failed: {e}. Falling back to HC1 robust.")

        # Save outputs
        coef_table = pd.DataFrame({
            "variable": res.params.index,
            "coef": res.params.values,
            "std_err": res.bse.values,
            "t": res.tvalues.values,
            "p_value": res.pvalues.values
        })
        table_path = os.path.join(ARTIFACT_DIR, "table_model1a.csv")
        coef_table.to_csv(table_path, index=False)

        with open(os.path.join(ARTIFACT_DIR, "model_summary.txt"), "w") as f:
            f.write(str(res.summary()))

        # Print a concise JSON-like summary to stdout for the orchestrator logs
        output = {
            "n_obs": int(res.nobs),
            "r_squared": float(getattr(res, "rsquared", np.nan)),
            "adj_r_squared": float(getattr(res, "rsquared_adj", np.nan)),
            "coefficients": {
                k: {
                    "coef": float(res.params[k]),
                    "std_err": float(res.bse[k]),
                    "p_value": float(res.pvalues[k])
                } for k in res.params.index
            },
            "table_file": table_path
        }
        print(json.dumps(output))

    except Exception as e:
        err_path = os.path.join(ARTIFACT_DIR, "error.log")
        with open(err_path, "w") as f:
            f.write("Error during execution:\n" + str(e) + "\n\nLogs:\n" + "\n".join(log_lines))
        # Also print to stdout so orchestrator captures it
        print(json.dumps({"error": str(e)}))


if __name__ == "__main__":
    main()
