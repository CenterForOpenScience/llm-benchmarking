# generator/execute/orchestrator.py
# Requires: pip install docker and Docker daemon running
from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

try:
    import docker  # type: ignore
except Exception:
    docker = None  # we throw a friendly error when used

# Data model
@dataclass
class PlanStep:
    name: str
    type: str  # "orchestrator" or "container"
    lang: str = ""                 # "r" | "python" | "bash"
    entry: Optional[str] = None    # filename declared in replication_info
    expected_artifacts: List[str] = field(default_factory=list)

@dataclass
class ExecutionPlan:
    plan_id: str
    steps: List[PlanStep]
    success_criteria: List[str] = field(default_factory=list)


# Planner
def _detect_lang_from_ext(filename: str) -> str:
    f = filename.lower()
    if f.endswith(".r"): return "r"
    if f.endswith(".py"): return "python"
    if f.endswith(".sh"): return "bash"
    return "bash"

def plan_from_replication_info(replication_info: Dict) -> ExecutionPlan:
    """Create: prepare-env → run declared entry file (no fallback)."""
    claim_id = (
        replication_info.get("original_study", {})
        .get("metadata", {})
        .get("original_paper_id", "study")
    )
    plan_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", claim_id)

    codebase = replication_info.get("original_study", {}).get("codebase", {}).get("files", {})
    if not codebase:
        raise ValueError("replication_info.original_study.codebase.files is empty; no entry script to run.")

    # Priority: .R, .py, .sh, else first
    keys = list(codebase.keys())
    ordered = (
        [k for k in keys if k.lower().endswith(".r")] +
        [k for k in keys if k.lower().endswith(".py")] +
        [k for k in keys if k.lower().endswith(".sh")] +
        [k for k in keys if not (k.lower().endswith((".r",".py",".sh")))]
    )
    entry = ordered[0]
    lang = _detect_lang_from_ext(entry)

    return ExecutionPlan(
        plan_id=plan_id,
        steps=[
            PlanStep(name="prepare-env", type="orchestrator"),
            PlanStep(name="run-analysis", type="container", lang=lang, entry=entry),
        ],
    )

# Orchestrator
class Orchestrator:
    def __init__(
        self,
        replication_json_path: str,
        study_path: str,
        image_name: str = "replication-exec",
        container_name: str = "replication-runner",
    ) -> None:
        """
        replication_json_path: path to the study's replication_info.json
        study_path: root folder that contains replication_data, code, etc.
        """
        self.replication_json_path = Path(replication_json_path).resolve()
        self.study_path = Path(study_path).resolve()
        self.image_name = image_name
        self.container_name = container_name

        # All build files & artifacts live under the study folder
        self.work_dir = self.study_path / "_runtime"
        self.art_dir = self.study_path / "_artifacts"
        self.dockerfile_path = self.work_dir / "Dockerfile"

        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.art_dir.mkdir(parents=True, exist_ok=True)

        self._cli = None

    # Docker client
    @property
    def cli(self):
        if docker is None:
            raise RuntimeError("The 'docker' package is not installed. Run: pip install docker")
        if self._cli is None:
            self._cli = docker.from_env()
        return self._cli

    def _spec(self) -> Dict:
        if not self.replication_json_path.exists():
            raise FileNotFoundError(f"replication_info.json not found at: {self.replication_json_path}")
        return json.loads(self.replication_json_path.read_text())

    # Dockerfile generation from json (base image must come from JSON)
    def generate_dockerfile(self) -> Dict:
        spec = self._spec()
        dspec = spec.get("docker_specs", {})
        base = dspec.get("base_image")
        if not base:
            raise ValueError("docker_specs.base_image is required in replication_info.json")

        r_pkgs = dspec.get("packages", {}).get("r", []) or []
        other  = dspec.get("packages", {}).get("other", []) or []
        py_pkgs = dspec.get("packages", {}).get("python", []) or []

        lines: List[str] = [f"FROM {base}"]

        if other:
            lines += [
                "RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y "
                + " ".join(other)
                + " && rm -rf /var/lib/apt/lists/*"
            ]

        if r_pkgs:
            rp = ",".join(f'"{p}"' for p in r_pkgs)
            lines.append(f"RUN R -q -e 'install.packages(c({rp}), repos=\"https://cloud.r-project.org\")'")

        if py_pkgs:
            lines += [
                "RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*",
                "RUN pip3 install " + " ".join(py_pkgs),
            ]

        # Create dirs as root, THEN switch user to avoid permission issues
        lines += [
            "WORKDIR /workspace",
            "RUN useradd -m runner && mkdir -p /app/data /app/artifacts /app/tmp && chown -R runner:runner /workspace /app",
            "USER runner",
            'CMD ["bash"]',
        ]

        self.dockerfile_path.write_text("\n".join(lines))
        return {"ok": True, "dockerfile": str(self.dockerfile_path)}

    def build_docker_image(self) -> Dict:
        # Build context is the per-study _runtime dir where the Dockerfile lives
        spec = self._spec()
        dspec = spec.get("docker_specs", {}) or {}
        py_pkgs = (dspec.get("packages", {}) or {}).get("python", []) or []

        # 1) respect explicit platform if present
        platform = dspec.get("platform")

        # 2) otherwise default on Apple Silicon/ARM hosts to amd64
        if not platform:
            host_arch = _pyplat.machine().lower()
            if host_arch in ("arm64", "aarch64"):
                platform = "linux/amd64"

        build_kwargs = dict(path=str(self.work_dir), tag=self.image_name, rm=True, pull=False)
        if platform:
            build_kwargs["platform"] = platform

        try:
            img, logs = self.cli.images.build(**build_kwargs)
            return {"ok": True, "image": self.image_name}
        except BuildError as e:
            # 3) Retry if the error explicitly complains about linux/arm64 manifest
            msg = str(e)
            if ("no matching manifest for linux/arm64" in msg.lower()
                or "no matching manifest for linux/arm64/v8" in msg.lower()):
                # force amd64 and retry once
                build_kwargs["platform"] = "linux/amd64"
                img, logs = self.cli.images.build(**build_kwargs)
                return {"ok": True, "image": self.image_name}
            raise

    def run_docker_container(
        self,
        mem_limit: Optional[str] = None,
        cpus: Optional[float] = None,
        read_only: bool = False,
        network_disabled: bool = False,
    ) -> Dict:
        spec = self._spec()

        # Build mounts keyed by CONTAINER PATH to prevent duplicate targets
        mounts_by_ctr: Dict[str, str] = {}

        # extra volumes from docker_specs
        for v in (spec.get("docker_specs", {}).get("volumes") or []):
            try:
                host, ctr = v.split(":", 1)
                ctr = ctr.strip()
                mounts_by_ctr[ctr] = os.path.abspath(host)
            except ValueError:
                pass  # skip malformed entries silently

        # If replication_data exists, FORCE it to /app/data (wins over anything else)
        repl_data = self.study_path / "replication_data"
        if repl_data.exists():
            mounts_by_ctr["/app/data"] = str(repl_data.resolve())

        # Always mount the study root and the artifacts dir (unique targets)
        mounts_by_ctr["/workspace"] = str(self.study_path)
        mounts_by_ctr["/app/artifacts"] = str(self.art_dir)

        # Convert to docker-py volumes dict {host: {"bind": ctr, "mode": "rw"}}
        volumes: Dict[str, Dict[str, str]] = {}
        for ctr, host in mounts_by_ctr.items():
            volumes[host] = {"bind": ctr, "mode": "rw"}

        kwargs = dict(
            image=self.image_name,
            name=self.container_name,
            command="sleep infinity",
            detach=True,
            working_dir="/workspace",
            volumes=volumes,
        )
        if mem_limit:
            kwargs["mem_limit"] = mem_limit
        if cpus:
            kwargs["nano_cpus"] = int(float(cpus) * 1e9)
        if read_only:
            kwargs["read_only"] = True
        if network_disabled:
            kwargs["network_disabled"] = True

        # Helpful debug: log final mounts
        print("[orchestrator] final mounts:")
        for ctr, host in mounts_by_ctr.items():
            print(f"  {host}  ->  {ctr}")

        container = self.cli.containers.run(**kwargs)
        return {"ok": True, "container": container.name}

    def stop_container(self) -> Dict:
        try:
            c = self.cli.containers.get(self.container_name)
            c.remove(force=True)
        except Exception:
            pass
        return {"ok": True}

    # container helpers
    def _container_path_exists(self, path: str) -> bool:
        c = self.cli.containers.get(self.container_name)
        parent = os.path.dirname(path) or "/"
        base = os.path.basename(path)
        exec_id = self.cli.api.exec_create(c.id, ["bash", "-lc", f'ls -1 {shq(parent)} || true'])
        out = self.cli.api.exec_start(exec_id, stream=False, demux=False)
        s = out.decode(errors="replace") if isinstance(out, (bytes, bytearray)) else str(out)
        return any(line.strip() == base for line in s.splitlines())

    def find_entry(self, entry: str) -> Optional[str]:
        candidates = [
            f"/workspace/{entry}",
            f"/workspace/replication_data/{entry}",
            f"/app/data/{entry}",
            f"/workspace/code/{entry}",
        ]
        for p in candidates:
            if self._container_path_exists(p):
                return p
        return None

    def _exec_file(self, container_path: str, lang: str) -> Dict:
        c = self.cli.containers.get(self.container_name)
        l = (lang or "").lower()
        if l == "r":
            cmd = ["Rscript", container_path]
        elif l == "python":
            cmd = ["python3", container_path]
        elif l == "bash":
            cmd = ["bash", container_path]
        else:
            return {"ok": False, "exit_code": 2, "stdout": "", "stderr": f"Unsupported lang: {lang}", "artifacts": []}

        exec_id = self.cli.api.exec_create(c.id, cmd, workdir="/workspace")
        output = self.cli.api.exec_start(exec_id, stream=False, demux=True, tty=False)
        exit_code = self.cli.api.exec_inspect(exec_id)["ExitCode"]

        stdout, stderr = output
        stdout = (stdout or b"").decode(errors="replace")
        stderr = (stderr or b"").decode(errors="replace")

        # list artifacts
        arts = []
        if self.art_dir.exists():
            try:
                arts = sorted([p.name for p in self.art_dir.iterdir() if p.is_file()])
            except Exception:
                pass

        return {
            "ok": exit_code == 0,
            "exit_code": exit_code,
            "stdout": stdout,
            "stderr": stderr,
            "artifacts": arts,
        }

    # one-shot plan executor
    def execute_plan(self, plan: ExecutionPlan) -> Dict:
        self.generate_dockerfile()
        self.build_docker_image()
        self.run_docker_container()

        results = {"plan_id": plan.plan_id, "steps": []}
        try:
            for step in plan.steps:
                if step.type == "orchestrator":
                    results["steps"].append({"name": step.name, "ok": True})
                    continue

                # container step: run the declared entry
                entry = step.entry
                if not entry:
                    results["steps"].append({
                        "name": step.name, "ok": False, "exit_code": 2,
                        "stdout": "", "stderr": "No entry file specified in replication_info.original_study.codebase.files",
                        "artifacts": [], "metrics": None, "entry": None,
                    })
                    continue

                found = self.find_entry(entry)
                if not found:
                    results["steps"].append({
                        "name": step.name, "ok": False, "exit_code": 2,
                        "stdout": "",
                        "stderr": f"Entry not found. Searched /workspace, /workspace/replication_data, /app/data, /workspace/code (entry='{entry}')",
                        "artifacts": [], "metrics": None, "entry": entry,
                    })
                    continue

                ran = self._exec_file(found, step.lang)
                results["steps"].append({
                    "name": step.name,
                    "ok": ran.get("ok", False),
                    "exit_code": ran.get("exit_code"),
                    "stdout": ran.get("stdout"),
                    "stderr": ran.get("stderr"),
                    "artifacts": ran.get("artifacts", []),
                    "metrics": None,
                    "entry": entry,
                    "resolved_path": found,
                    "lang": step.lang,
                })

            # overall ok = all container steps ok (excluding prepare-env)
            results["ok"] = all(s.get("ok", False) for s in results["steps"] if s["name"] != "prepare-env")
            return results
        finally:
            self.stop_container()

# Utilities
def shq(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"

def _cli():
    parser = argparse.ArgumentParser(description="Orchestrator CLI")
    parser.add_argument("--replication_json", required=True, help="Path to replication_info.json")
    parser.add_argument("--study_path", required=True, help="Path to study root directory")
    sub = parser.add_subparsers(dest="cmd", required=True)
    sub.add_parser("plan-run")
    args = parser.parse_args()

    orch = Orchestrator(replication_json_path=args.replication_json, study_path=args.study_path)
    info = json.loads(Path(args.replication_json).read_text())
    plan = plan_from_replication_info(info)
    print(json.dumps(orch.execute_plan(plan), indent=2))


if __name__ == "__main__":
    _cli()
