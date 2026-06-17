import re

# Expected columns based on the R analysis script
EXPECTED_VARS = [
    "trstprl_rev", "imm_concern", "happy_rev", "stflife_rev", "sclmeet_rev",
    "distrust_soc", "stfeco_rev", "hincfel", "stfhlth_rev", "stfedu_rev",
    "vote_gov", "vote_frparty", "lrscale", "hhinc_std", "agea", "educ", "female",
    "vote_share_fr", "socexp", "lt_imm_cntry", "wgi", "gdppc", "unemp", "cntry", "pspwght"
]


def infer_mapping(columns):
    """Return a mapping from expected var names to actual dataset columns (case-insensitive).
    If a var is not found, it is mapped to None.
    """
    cols = list(columns)
    mapping = {}
    lower_map = {c.lower(): c for c in cols}
    for var in EXPECTED_VARS:
        if var in cols:
            mapping[var] = var
        elif var.lower() in lower_map:
            mapping[var] = lower_map[var.lower()]
        else:
            mapping[var] = None
    return mapping


def missing_vars(mapping, required=None):
    req = EXPECTED_VARS if required is None else required
    return [v for v in req if mapping.get(v) is None]
