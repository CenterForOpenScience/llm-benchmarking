import os, json
from typing import Any, Dict
from .io import read_json

def load_original_schema(templates_dir: str) -> Dict[str, Any]:
    path = os.path.join(templates_dir, "post_registration_schema.json")
    schema = read_json(path)
    if not schema or "original_study" not in schema:
        raise FileNotFoundError(f"Could not load original_study schema from {path}")
    return schema

def blank_from_schema(schema: Any) -> Any:
    if isinstance(schema, dict):
        return {k: blank_from_schema(v) for k, v in schema.items()}
    if isinstance(schema, list):
        return []
    return None

def build_combined_template(templates_dir: str) -> Dict[str, Any]:
    base = load_original_schema(templates_dir)
    original_blank = blank_from_schema(base).get("original_study", {})

    replication_schema = {
        "hypothesis": None,
        "study_type": None,
        "data_plan": {
            "dataset_identifier": None,
            "source_type": None,
            "wave_or_subset": None,
            "sample_size": None,
            "unit_of_analysis": None,
            "access_details": None,
            "notes": None
        },
        "planned_method": {
            "steps": [],
            "models": None,
            "outcome_variable": None,
            "independent_variables": None,
            "control_variables": [],
            "tools_software": None
        },
        "planned_estimation_and_test": {
            "estimation": None,
            "test": None,
            "missing_data_handling": "Listwise deletion unless otherwise specified.",
            "multiple_testing_policy": "None; if >1 primary outcome, apply BH-FDR (q=0.10).",
            "inference_criteria": "Two-sided α=0.05; focal signs: β1>0 (violence), β2<0 (violence²)."
        }
    }
    return {"original_study": original_blank, "Replication": replication_schema}

# Deterministic fallback
def first_nonnull(*vals):
    for v in vals:
        if v not in (None, "", [], {}):
            return v
    return None


def fallback_build(schema: Dict[str, Any], inputs: Dict[str, Any]) -> Dict[str, Any]:
    out = json.loads(json.dumps(schema))  # deep copy
    pr = inputs.get("post_registration") or {}
    pr_os = pr.get("original_study", {})

    if pr_os:
        out["original_study"]["claim"]["hypotheses"] = first_nonnull(
            "; ".join(pr_os.get("claim", {}).get("hypotheses", []) or []),
            pr_os.get("claim", {}).get("hypothesis")
        )
        out["original_study"]["claim"]["hypotheses_location"] = pr_os.get("claim", {}).get("hypotheses_location")
        out["original_study"]["claim"]["statement"] = pr_os.get("claim", {}).get("statement")
        out["original_study"]["claim"]["statement_location"] = pr_os.get("claim", {}).get("statement_location")
        out["original_study"]["claim"]["study_type"] = first_nonnull(
            pr_os.get("claim", {}).get("study_type"),
            pr_os.get("study_type")
        )
        out["original_study"]["data"]["source"] = pr_os.get("data", {}).get("source")
        out["original_study"]["data"]["wave_or_subset"] = pr_os.get("data", {}).get("wave_or_subset")
        out["original_study"]["data"]["sample_size"] = pr_os.get("data", {}).get("sample_size")
        out["original_study"]["data"]["unit_of_analysis"] = pr_os.get("data", {}).get("unit_of_analysis")
        out["original_study"]["data"]["access_details"] = pr_os.get("data", {}).get("access_details")
        out["original_study"]["data"]["notes"] = pr_os.get("data", {}).get("notes")
        out["original_study"]["method"]["description"] = pr_os.get("method", {}).get("description")
        out["original_study"]["method"]["steps"] = pr_os.get("method", {}).get("steps", [])
        out["original_study"]["method"]["models"] = pr_os.get("method", {}).get("models")
        out["original_study"]["method"]["outcome_variable"] = pr_os.get("method", {}).get("outcome_variable")
        out["original_study"]["method"]["independent_variables"] = pr_os.get("method", {}).get("independent_variables")
        out["original_study"]["method"]["control_variables"] = pr_os.get("method", {}).get("control_variables", [])
        out["original_study"]["method"]["tools_software"] = pr_os.get("method", {}).get("tools_software")
        if "results" in pr_os:
            out["original_study"]["results"] = pr_os.get("results", {})
        out["original_study"]["metadata"]["original_paper_id"] = pr_os.get("metadata", {}).get("original_paper_id")
        out["original_study"]["metadata"]["original_paper_title"] = pr_os.get("metadata", {}).get("original_paper_title")
        out["original_study"]["metadata"]["original_paper_code"] = pr_os.get("metadata", {}).get("original_paper_code")
        out["original_study"]["metadata"]["original_paper_data"] = pr_os.get("metadata", {}).get("original_paper_data")

    ri = inputs.get("replication_info") or {}
    ri_os = ri.get("original_study", {})
    ri_rep_ds = ri.get("replication_datasets") or []

    out["Replication"]["hypothesis"] = (
        "At the district level, fraud increases with violence at low levels and decreases at high levels: "
        "fraud = β0 + β1·violence + β2·violence² + γ'X + ε with β1>0, β2<0."
    )
    out["Replication"]["study_type"] = first_nonnull(
        ri_os.get("claim", {}).get("study_type"),
        "Observational"
    )

    dp = out["Replication"]["data_plan"]
    dp["dataset_identifier"] = ri_rep_ds[0]["name"] if ri_rep_ds else None
    dp["source_type"] = "Election results + conflict incident database"
    dp["wave_or_subset"] = first_nonnull(ri_os.get("data", {}).get("wave_or_subset"), "TBD")
    dp["sample_size"] = ri_os.get("data", {}).get("sample_size")
    dp["unit_of_analysis"] = first_nonnull(ri_os.get("data", {}).get("unit_of_analysis"), "District")
    dp["access_details"] = ri_os.get("data", {}).get("access_details")
    dp["notes"] = "Derived from Stage-1 output and replication_info; fill dataset version/URL during Execute."

    pm = out["Replication"]["planned_method"]
    pm["steps"] = [
        "Load replication dataset and harmonize district identifiers.",
        "Construct fraud measure as in original (last-digit test or recount-based index).",
        "Create violence variables for the election window and pre-election period; compute violence².",
        "Merge covariates (e.g., centers closed %, electrification, expenditure, distance to Kabul, elevation).",
        "Estimate models: fraud ~ violence + violence² + controls.",
        "Extract coefficients, SEs, p-values; compute turning point and visualize inverted-U."
    ]
    pm["models"] = first_nonnull(
        ri_os.get("method", {}).get("models"),
        "Logit for binary fraud; OLS for continuous fraud index."
    )
    pm["outcome_variable"] = first_nonnull(
        ri_os.get("method", {}).get("outcome_variable"),
        "Election fraud (forensic last-digit / recount-based)."
    )
    pm["independent_variables"] = first_nonnull(
        ri_os.get("method", {}).get("independent_variables"),
        "Violence level and Violence squared"
    )
    pm["control_variables"] = first_nonnull(
        ri_os.get("method", {}).get("control_variables"),
        ["Percentage of centers closed", "Electrification", "Per-capita expenditure", "Distance from Kabul", "Elevation"]
    )
    tool = None
    if "codebase" in ri and any(name.endswith(".do") for name in (ri["codebase"].get("files") or {})):
        tool = "Stata"
    base_img = (ri.get("docker_specs") or {}).get("base_image")
    if base_img and "stata" in str(base_img).lower():
        tool = "Stata"
    pm["tools_software"] = tool or "Stata or Python/R (to be finalized)."

    pet = out["Replication"]["planned_estimation_and_test"]
    pet["estimation"] = "Coefficients on violence (β1) and violence² (β2) in the fraud regression."
    pet["test"] = "Two-sided z/t-tests for β1>0 and β2<0; joint Wald test for inverted-U."
    return out
