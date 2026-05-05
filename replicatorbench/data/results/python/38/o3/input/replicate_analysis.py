"""
Replication analysis for McCarter et al. (2010) focal hypothesis.
This script loads the cleaned dataset and estimates whether contributions
(`cont`) are lower when the bonus structure includes the possibility of
losses (`losses == 1`) compared to gains-only (`losses == 0`).

We follow the planned specification which fits a linear mixed-effects
model with random intercepts for participant (`id`) and an additional
variance component for session (`sessioncode`).

All file paths are hard-coded to `/app/data` because the orchestrator
mounts that directory into the Docker container.
"""
from pathlib import Path
import pandas as pd
from statsmodels.regression.mixed_linear_model import MixedLM

# ---------------------------------------------------------------------
# Constants / paths
# ---------------------------------------------------------------------
DATA_PATH = Path("/app/data/McCarter_OrgBehavior_2010_pILK data CLEAN.csv")
RESULT_PATH = Path("/app/data/replication_results.csv")


def main() -> None:
    """Run the replication analysis and save results."""

    # -----------------------------------------------------------------
    # Load and preprocess data
    # -----------------------------------------------------------------
    df = pd.read_csv(DATA_PATH)

    # Subset to required columns and drop any missing values (none expected)
    cols_needed = ["cont", "losses", "id", "sessioncode"]
    df_model = df[cols_needed].dropna(how="any")

    # -----------------------------------------------------------------
    # Specify and fit the mixed-effects model
    #   cont ~ losses           (fixed effects)
    #   random intercept for id (groups)
    #   variance component for sessioncode
    # -----------------------------------------------------------------
    md = MixedLM.from_formula(
        formula="cont ~ losses",
        groups="id",
        vc_formula={"session": "0 + C(sessioncode)"},
        data=df_model,
    )
    mdf = md.fit(reml=False)

    # -----------------------------------------------------------------
    # Output
    # -----------------------------------------------------------------
    print(mdf.summary())

    # Save coefficients and p-values for downstream checks
    out = pd.DataFrame({"coef": mdf.params, "pval": mdf.pvalues})
    out.to_csv(RESULT_PATH, index_label="term")
    print(f"Results saved to {RESULT_PATH}")


if __name__ == "__main__":
    main()
