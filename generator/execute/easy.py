# generator/execute/easy.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from .orchestrator import Orchestrator, plan_from_replication_info


def run_execute_easy(study_path: str, logger: Optional[object] = None) -> dict:
    study_dir = Path(study_path).resolve()
    rep_info_path = study_dir / "replication_info.json"

    if not rep_info_path.exists():
        raise FileNotFoundError(f"replication_info.json not found at: {rep_info_path}")

    if logger:
        logger.info(f"[execute] using replication info: {rep_info_path}")
        logger.info("[execute] starting plan execution...")

    # read once (for planning); the orchestrator will also read from the same path
    rep_info = json.loads(rep_info_path.read_text())

    # construct orchestrator with the REAL file path (no temp/merged copy)
    orch = Orchestrator(
        replication_json_path=str(rep_info_path),
        study_path=str(study_dir),
    )

    plan = plan_from_replication_info(rep_info)
    result = orch.execute_plan(plan)

    if logger:
        logger.info(f"[execute] done. ok={result.get('ok')}")

    out_path = study_dir / "execution_result.json"
    out_path.write_text(json.dumps(result, indent=2))
    if logger:
        logger.info(f"[execute] wrote results → {out_path}")

    return result
