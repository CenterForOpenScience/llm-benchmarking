import os
import json
import pandas as pd
import numpy as np
import statsmodels.api as sm

# All IO must use /app/data
DATA_PATH = "/app/data/finaldata_noNA.csv"
OUTPUT_SUMMARY_TXT = "/app/data/replication_results_affluence_trade.txt"
OUTPUT_KEY_JSON = "/app/data/replication_key_results.json"


def main():
    # Load data
    df = pd.read_csv(DATA_PATH)

    # Basic checks
    required_cols = ["country", "year", "gdp", "pop", "totalimport", "totalexport", "unemp"]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Required column '{c}' not found in dataset.")

    # Construct variables
    # National affluence: log GDP per capita (to align with common practice in macro panel regressions)
    df = df.copy()
    df["gdp_pc"] = df["gdp"] / df["pop"].replace({0: np.nan})
    df["affluence"] = np.log(df["gdp_pc"])  # natural log

    # Trade intensity: imports and exports as shares of GDP (approximate for missing South disaggregation)
    df["import_share_gdp"] = df["totalimport"] / df["gdp"].replace({0: np.nan})
    df["export_share_gdp"] = df["totalexport"] / df["gdp"].replace({0: np.nan})

    # Sort and lag by country
    df = df.sort_values(["country", "year"]).reset_index(drop=True)
    df["import_share_gdp_l1"] = df.groupby("country")["import_share_gdp"].shift(1)
    df["export_share_gdp_l1"] = df.groupby("country")["export_share_gdp"].shift(1)
    df["unemp_l1"] = df.groupby("country")["unemp"].shift(1)

    # Drop rows with missing values in key variables post-lag
    model_df = df.dropna(subset=["affluence", "import_share_gdp_l1", "export_share_gdp_l1", "unemp_l1", "country", "year"]).copy()

    # Two-way fixed effects via country and year dummies
    country_dummies = pd.get_dummies(model_df["country"], prefix="cty", drop_first=True, dtype=float)
    year_dummies = pd.get_dummies(model_df["year"].astype(int), prefix="yr", drop_first=True, dtype=float)

    X_base = model_df[["import_share_gdp_l1", "export_share_gdp_l1", "unemp_l1"]].astype(float)
    X = pd.concat([X_base, country_dummies, year_dummies], axis=1)
    X = sm.add_constant(X).astype(float)
    y = model_df["affluence"].astype(float).values

    # Fit OLS with cluster-robust SEs by country
    model = sm.OLS(y, X)
    result = model.fit(cov_type="cluster", cov_kwds={"groups": model_df["country"]})

    # Prepare outputs
    # Save full summary
    with open(OUTPUT_SUMMARY_TXT, "w") as f:
        f.write(result.summary().as_text())
        f.write("\n\nNotes:\n")
        f.write("- Dependent variable: log(GDP per capita) as proxy for national affluence.\n")
        f.write("- Key regressor: lagged imports as share of GDP (total imports due to lack of South-specific decomposition).\n")
        f.write("- Fixed effects: country and year dummies. Clustered SEs by country.\n")

    # Key coefficient of interest: import_share_gdp_l1
    coef_name = "import_share_gdp_l1"
    if coef_name in result.params.index:
        key = {
            "dependent_variable": "affluence_log_gdp_per_capita",
            "independent_variable": coef_name,
            "coefficient": float(result.params[coef_name]),
            "std_error": float(result.bse[coef_name]),
            "t_value": float(result.tvalues[coef_name]),
            "p_value": float(result.pvalues[coef_name]),
            "n_obs": int(model_df.shape[0]),
            "n_countries": int(model_df["country"].nunique()),
            "years_range": [int(model_df["year"].min()), int(model_df["year"].max())],
            "cov_type": "cluster_country",
            "controls": ["export_share_gdp_l1", "unemp_l1"],
            "fixed_effects": ["country", "year"],
        }
    else:
        key = {
            "error": "Coefficient import_share_gdp_l1 not found in results.",
            "available_params": list(map(str, result.params.index))
        }

    with open(OUTPUT_KEY_JSON, "w") as f:
        json.dump(key, f, indent=2)

    # Print small console message for runtime logs
    print(json.dumps({"status": "completed", "output_summary": OUTPUT_SUMMARY_TXT, "output_key": OUTPUT_KEY_JSON}, indent=2))


if __name__ == "__main__":
    main()
