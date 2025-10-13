from __future__ import annotations
import json
from pathlib import Path
from typing import Optional, Dict, Any
from .orchestrator import Orchestrator, plan_from_replication_info, ExecutionPlan, PlanStep

def _paths(study_path: str):
    study_dir = Path(study_path).resolve()
    rep_info = study_dir / "replication_info.json"
    out_path = study_dir / "execution_result.json"
    return study_dir, rep_info, out_path

def _paths(study_path: str):
    study_dir = Path(study_path).resolve()
    rep_info = study_dir / "replication_info.json"
    out_path = study_dir / "execution_result.json"
    return study_dir, rep_info, out_path

def orchestrator_generate_dockerfile(study_path: str) -> str:
    """Create Dockerfile from replication_info.json → _runtime/Dockerfile"""
    study_dir, rep_info, _ = _paths(study_path)
    orch = Orchestrator(replication_json_path=str(rep_info), study_path=str(study_dir))
    res = orch.generate_dockerfile()
    return json.dumps(res)

def orchestrator_build_image(study_path: str) -> str:
    """docker build (respects docker_specs.platform if present)"""
    study_dir, rep_info, _ = _paths(study_path)
    orch = Orchestrator(replication_json_path=str(rep_info), study_path=str(study_dir))
    res = orch.build_docker_image()
    return json.dumps(res)

def orchestrator_run_container(study_path: str,
                               mem_limit: Optional[str] = None,
                               cpus: Optional[float] = None,
                               read_only: bool = False,
                               network_disabled: bool = False) -> str:
    """docker run with mounts (/workspace, /app/data, /app/artifacts)"""
    study_dir, rep_info, _ = _paths(study_path)
    orch = Orchestrator(replication_json_path=str(rep_info), study_path=str(study_dir))
    res = orch.run_docker_container(mem_limit=mem_limit, cpus=cpus,
                                    read_only=read_only, network_disabled=network_disabled)
    return json.dumps(res)

def orchestrator_plan(study_path: str) -> str:
    """Return the computed ExecutionPlan (as JSON) from replication_info.json"""
    study_dir, rep_info, _ = _paths(study_path)
    info = json.loads(rep_info.read_text())
    plan = plan_from_replication_info(info)
    # make it json-serializable
    out = {
        "plan_id": plan.plan_id,
        "steps": [{"name": s.name, "type": s.type, "lang": s.lang, "entry": s.entry,
                   "expected_artifacts": s.expected_artifacts} for s in plan.steps],
        "success_criteria": plan.success_criteria,
    }
    return json.dumps(out)

def orchestrator_execute_entry(study_path: str) -> str:
    """
    Assumes a container is running. Resolves the declared entry from the plan,
    executes it inside the container, aggregates results, writes execution_result.json,
    and returns that JSON as a string.
    """
    study_dir, rep_info, out_path = _paths(study_path)
    info = json.loads(rep_info.read_text())
    orch = Orchestrator(replication_json_path=str(rep_info), study_path=str(study_dir))
    plan = plan_from_replication_info(info)

    # build a results shell similar to Orchestrator.execute_plan()
    results: Dict[str, Any] = {"plan_id": plan.plan_id, "steps": []}

    # "prepare-env" was already done by earlier steps; mark as ok=True for traceability
    results["steps"].append({"name": "prepare-env", "ok": True})

    # find the container step
    step: Optional[PlanStep] = next((s for s in plan.steps if s.type == "container"), None)
    if not step or not step.entry:
        results["steps"].append({
            "name": "run-analysis", "ok": False, "exit_code": 2,
            "stdout": "", "stderr": "No entry file specified.",
            "artifacts": [], "metrics": None, "entry": step.entry if step else None,
        })
        results["ok"] = False
        out_path.write_text(json.dumps(results, indent=2))
        return json.dumps(results)

    found = orch.find_entry(step.entry)
    if not found:
        results["steps"].append({
            "name": step.name, "ok": False, "exit_code": 2,
            "stdout": "",
            "stderr": f"Entry not found. Searched /workspace, /workspace/replication_data, /app/data, /workspace/code (entry='{step.entry}')",
            "artifacts": [], "metrics": None, "entry": step.entry,
        })
        results["ok"] = False
        out_path.write_text(json.dumps(results, indent=2))
        return json.dumps(results)

    ran = orch._exec_file(found, step.lang)
    results["steps"].append({
        "name": step.name,
        "ok": ran.get("ok", False),
        "exit_code": ran.get("exit_code"),
        "stdout": ran.get("stdout"),
        "stderr": ran.get("stderr"),
        "artifacts": ran.get("artifacts", []),
        "metrics": None,
        "entry": step.entry,
        "resolved_path": found,
        "lang": step.lang,
    })
    # overall ok = just the container step’s ok
    results["ok"] = all(s.get("ok", False) for s in results["steps"] if s["name"] != "prepare-env")

    out_path.write_text(json.dumps(results, indent=2))
    return json.dumps(results)

def orchestrator_stop_container(study_path: str) -> str:
    """Stop & remove the container (idempotent)"""
    study_dir, rep_info, _ = _paths(study_path)
    orch = Orchestrator(replication_json_path=str(rep_info), study_path=str(study_dir))
    return json.dumps(orch.stop_container())


def orchestrator_preview_entry(study_path: str) -> str:
    """
    Returns the resolved inside-container path and the exact command that would be executed.
    Does NOT execute anything.
    """
    study_dir, rep_info, _ = _paths(study_path)
    info = json.loads(rep_info.read_text())
    orch = Orchestrator(replication_json_path=str(rep_info), study_path=str(study_dir))
    plan = plan_from_replication_info(info)

    # find the container step
    step = next((s for s in plan.steps if s.type == "container"), None)
    if not step or not step.entry:
        return json.dumps({"ok": False, "error": "No container step or entry file specified."})

    found = orch.find_entry(step.entry)
    if not found:
        return json.dumps({
            "ok": False,
            "error": f"Entry not found. Searched /workspace, /workspace/replication_data, /app/data, /workspace/code (entry='{step.entry}')",
            "entry": step.entry,
        })

    l = (step.lang or "").lower()
    if l == "r":
        cmd = ["Rscript", found]
    elif l == "python":
        cmd = ["python3", found]
    elif l == "bash":
        cmd = ["bash", found]
    else:
        return json.dumps({"ok": False, "error": f"Unsupported lang: {step.lang}", "entry": step.entry, "resolved_path": found})

    return json.dumps({
        "ok": True,
        "plan_id": plan.plan_id,
        "lang": step.lang,
        "entry": step.entry,
        "resolved_path": found,
        "container_command": cmd,           # JSON array of argv
        "command_pretty": " ".join(cmd),   # human-friendly string
    })
