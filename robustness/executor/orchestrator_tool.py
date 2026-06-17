from __future__ import annotations
import io
import json
import os
import re
import platform as _pyplat
import tarfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from replicatorbench.info_extractor.file_utils import check_long_logs
import tiktoken

try:
    from docker.errors import BuildError, APIError
    import docker  # type: ignore
except Exception:
    docker = None
    BuildError = Exception
    APIError = Exception

# Planning data structures
@dataclass
class PlanStep:
    name: str
    type: str  # "orchestrator" or "container"
    lang: str = ""                 # "r" | "python" | "bash"
    entry: Optional[str] = None    # filename declared in analysis_info
    expected_artifacts: List[str] = field(default_factory=list)

@dataclass
class ExecutionPlan:
    plan_id: str
    steps: List[PlanStep]
    success_criteria: List[str] = field(default_factory=list)

# Helpers & constants
DEFAULT_IMAGE_NAME = "analysis-exec"
DEFAULT_CONTAINER_NAME = "analysis-runner"
COPIED_OUTPUTS_DIRNAME = "_copied_outputs"

def _detect_lang_from_ext(filename: str) -> str:
    f = filename.lower()
    if f.endswith(".r"): return "r"
    if f.endswith(".py"): return "python"
    if f.endswith(".sh"): return "bash"
    return "bash"

def _require_docker():
    if docker is None:
        raise RuntimeError("The 'docker' package is not installed. Run: pip install docker")
    return docker.from_env()

def _paths(study_path: str) -> Tuple[Path, Path, Path, Path, Path]:
    study_dir = Path(study_path).resolve()
    runtime_dir = study_dir / "_runtime"
    art_dir = study_dir / "_artifacts"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    art_dir.mkdir(parents=True, exist_ok=True)
    return study_dir, runtime_dir, art_dir, (runtime_dir / "Dockerfile"), (study_dir / "analysis_info.json")

def _copied_outputs_dir(study_path: str) -> Path:
    study_dir = Path(study_path).resolve()
    copied_dir = study_dir / COPIED_OUTPUTS_DIRNAME
    copied_dir.mkdir(parents=True, exist_ok=True)
    return copied_dir

def _copy_container_dir(container_name: str, container_dir: str, host_dir: Path) -> None:
    if not _container_path_exists(container_name, container_dir):
        return

    cli = _require_docker()
    container = cli.containers.get(container_name)
    stream, _ = container.get_archive(container_dir)
    archive = io.BytesIO()
    for chunk in stream:
        archive.write(chunk)
    if archive.tell() == 0:
        return

    archive.seek(0)
    root_name = Path(container_dir).name

    with tarfile.open(fileobj=archive) as tar:
        for member in tar.getmembers():
            member_path = Path(member.name)
            parts = member_path.parts
            if parts and parts[0] == root_name:
                rel_parts = parts[1:]
            else:
                rel_parts = parts

            if not rel_parts or any(part == ".." for part in rel_parts):
                continue

            dest = host_dir.joinpath(*rel_parts)
            if member.isdir():
                dest.mkdir(parents=True, exist_ok=True)
                continue

            if not member.isfile():
                continue

            extracted = tar.extractfile(member)
            if extracted is None:
                continue

            dest.parent.mkdir(parents=True, exist_ok=True)
            with dest.open("wb") as f:
                f.write(extracted.read())

def _copy_container_outputs(study_path: str, container_name: str) -> None:
    copied_root = _copied_outputs_dir(study_path)
    for container_dir, local_name in (
        ("/app/tmp", "app_tmp"),
        ("/tmp/artifacts", "tmp_artifacts"),
    ):
        _copy_container_dir(container_name, container_dir, copied_root / local_name)

def _list_local_output_files(study_path: str) -> List[str]:
    study_dir, _, art_dir, _, _ = _paths(study_path)
    arts: List[str] = []

    if art_dir.exists():
        try:
            arts.extend(sorted([p.name for p in art_dir.iterdir() if p.is_file()]))
        except Exception:
            pass

    copied_root = study_dir / COPIED_OUTPUTS_DIRNAME
    if copied_root.exists():
        try:
            arts.extend(
                sorted(
                    path.relative_to(study_dir).as_posix()
                    for path in copied_root.rglob("*")
                    if path.is_file()
                )
            )
        except Exception:
            pass

    return arts

def _read_spec(study_path: str) -> Dict:
    study_dir, _, _, _, rep_info = _paths(study_path)
    if not rep_info.exists():
        raise FileNotFoundError(f"analysis_info.json not found at: {rep_info}")
    return json.loads(rep_info.read_text())

def shq(s: str) -> str:
    return "'" + s.replace("'", "'\"'\"'") + "'"

# Planner
def plan_from_analysis_info(analysis_info: Dict, code_mode) -> ExecutionPlan:
    claim_id = (
        analysis_info.get("analysis_study", {})
        .get("metadata", {})
        .get("original_paper_id", "study")
    )
    plan_id = re.sub(r"[^a-zA-Z0-9_-]+", "-", claim_id)

    codebase = analysis_info.get("codebase", {}).get("files", {})
    if not codebase:
        # Fallback if codebase files empty but entry script might exist physically
        # We'll rely on agent to provide valid entry, or raise error later
        ordered = ["main.py"]
        lang = "python"
    else:
        # Priority: .R, .py, .sh, else first
        keys = list(codebase.keys())
        if code_mode == "python":
            ordered = (
                [k for k in keys if k.lower().endswith(".py")] +
                # [k for k in keys if k.lower().endswith(".r")] +
                [k for k in keys if k.lower().endswith(".sh")] +
                [k for k in keys if not (k.lower().endswith((".r",".py",".sh")))]
            )
        else:
            ordered = (
                # [k for k in keys if k.lower().endswith(".py")] +
                [k for k in keys if k.lower().endswith(".r")] +
                [k for k in keys if k.lower().endswith(".sh")] +
                [k for k in keys if not (k.lower().endswith((".r",".py",".sh")))]
            )
    
    steps=[
            PlanStep(name="prepare-env", type="orchestrator")
    ]
    for entry_id, entry in enumerate(ordered):
        lang = _detect_lang_from_ext(entry)
        steps.append(PlanStep(name=f"run-analysis-{entry_id}", type="container", lang=lang, entry=entry))
    
    return ExecutionPlan(
        plan_id=plan_id,
        steps=steps  
    )

def _get_docker_specs(spec: Dict) -> Dict:
    d = spec.get("docker_specs")
    if d: return d
    return spec.get("docker_specs", {}) or {}

# Tools
def orchestrator_generate_dockerfile(study_path: str) -> str:
    try:
        spec = _read_spec(study_path)
        dspec = _get_docker_specs(spec)

        base = dspec.get("base_image")
        if not base:
            # Smart default fallback
            base = "python:3.9-slim"
            # raise ValueError(f"docker_specs.base_image is required in {study_path}analysis_info.json")

        r_pkgs = (dspec.get("packages", {}) or {}).get("r", []) or []
        other  = (dspec.get("packages", {}) or {}).get("other", []) or []
        py_pkgs = (dspec.get("packages", {}) or {}).get("python", []) or []

        _, runtime_dir, _, dockerfile_path, _ = _paths(study_path)
        lines: List[str] = [f"FROM {base}"]

        # Basic tools
        lines.append("RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends git wget ca-certificates && rm -rf /var/lib/apt/lists/*")

        if other:
            lines += [
                "RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y "
                + " ".join(other)
                + " && rm -rf /var/lib/apt/lists/*"
            ]

        if r_pkgs:
            # Check if R is installed in base, if not install it
            #lines.append("RUN command -v R || (apt-get update && apt-get install -y r-base)")
            lines.append("RUN apt-get update && apt-get install -y --no-install-recommends r-base r-base-dev && rm -rf /var/lib/apt/lists/*")
            lines.append("RUN command -v Rscript && Rscript --version")
            lines.append('RUN Rscript --version || true')
            rp = ",".join(f'"{p}"' for p in r_pkgs)
            lines.append(f"RUN R -q -e 'install.packages(c({rp}), repos=\"https://cloud.r-project.org\")'")

        if py_pkgs:
            lines.append("RUN command -v python3 || (apt-get update && apt-get install -y python3 python3-pip)")
            lines.append("RUN pip3 install --no-cache-dir " + " ".join(py_pkgs))

        lines += [
            "WORKDIR /workspace",
            "RUN useradd -m runner && mkdir -p /app/data /app/artifacts /app/tmp && chown -R runner:runner /workspace /app",
            "USER runner",
            'CMD ["bash"]',
        ]

        dockerfile_path.write_text("\n".join(lines))
        return json.dumps({"ok": True, "dockerfile": str(dockerfile_path), "content": "\n".join(lines)})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})

def orchestrator_build_image(study_path: str, image_name: str = DEFAULT_IMAGE_NAME) -> str:
    """
    Builds Docker image. Captures logs on failure for agent debugging.
    """
    try:
        _ = _read_spec(study_path)
        spec = _read_spec(study_path)
        dspec = spec.get("docker_specs", {}) or {}
        platform = dspec.get("platform")
        if not platform:
            host_arch = _pyplat.machine().lower()
            if host_arch in ("arm64", "aarch64"):
                platform = "linux/amd64"

        cli = _require_docker()
        _, runtime_dir, _, _, _ = _paths(study_path)

        build_kwargs = dict(path=str(runtime_dir), tag=image_name, rm=True, pull=False)
        if platform:
            build_kwargs["platform"] = platform

        img, logs = cli.images.build(**build_kwargs)
        (runtime_dir / "image_name.txt").write_text(image_name)
        return json.dumps({"ok": True, "image": image_name})

    except (BuildError, APIError) as e:
        # Extract build logs to help agent debug
        log_lines = []
        if hasattr(e, 'build_log'):
            for chunk in e.build_log:
                if 'stream' in chunk:
                    log_lines.append(chunk['stream'].strip())
                elif 'error' in chunk:
                    log_lines.append(f"ERROR: {chunk['error']}")
        
        full_log = "\n".join(log_lines[-20:]) # Last 20 lines usually contain the error
        return json.dumps({
            "ok": False, 
            "error": "Docker build failed. See 'build_log' for details.", 
            "build_log": full_log,
            "exception": str(e)
        })
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})

def orchestrator_run_container(
    study_path: str,
    mem_limit: Optional[str] = None,
    cpus: Optional[float] = None,
    read_only: bool = False,
    network_disabled: bool = False,
    image_name: str = DEFAULT_IMAGE_NAME,
    container_name: str = DEFAULT_CONTAINER_NAME,
) -> str:
    try:
        cli = _require_docker()
        spec = _read_spec(study_path)
        study_dir, _, art_dir, _, _ = _paths(study_path)
        image_file = study_dir / "_runtime" / "image_name.txt"
        if image_file.exists():
            image_name = image_file.read_text().strip()

        try:
            old = cli.containers.get(container_name)
            old.remove(force=True)
        except Exception:
            pass

        mounts_by_ctr: Dict[str, str] = {}
        # Extra volumes
        for v in (spec.get("docker_specs", {}).get("volumes") or []):
            try:
                host, ctr = v.split(":", 1)
                mounts_by_ctr[ctr.strip()] = str(Path(host).resolve())
            except ValueError:
                pass

        repl_data = study_dir / "data"
        if repl_data.exists():
            mounts_by_ctr["/app/data"] = str(repl_data.resolve())

        mounts_by_ctr["/workspace"] = str(study_dir)
        mounts_by_ctr["/app/artifacts"] = str(art_dir)

        volumes: Dict[str, Dict[str, str]] = {}
        for ctr, host in mounts_by_ctr.items():
            volumes[host] = {"bind": ctr, "mode": "rw"}

        kwargs = dict(
            image=image_name,
            name=container_name,
            command="sleep infinity",
            detach=True,
            working_dir="/workspace",
            volumes=volumes,
        )
        if mem_limit: kwargs["mem_limit"] = mem_limit
        if cpus: kwargs["nano_cpus"] = int(float(cpus) * 1e9)
        if read_only: kwargs["read_only"] = True
        if network_disabled: kwargs["network_disabled"] = True

        container = cli.containers.run(**kwargs)
        return json.dumps({"ok": True, "container": container.name})
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})

def orchestrator_stop_container(study_path: str) -> str:
    cli = _require_docker()
    try:
        c = cli.containers.get(DEFAULT_CONTAINER_NAME)
        c.remove(force=True)
    except Exception:
        pass
    return json.dumps({"ok": True})

# Container exec helpers
def _container_path_exists(container_name: str, path: str) -> bool:
    cli = _require_docker()
    c = cli.containers.get(container_name)
    parent = os.path.dirname(path) or "/"
    base = os.path.basename(path)
    try:
        exec_id = cli.api.exec_create(c.id, ["bash", "-lc", f'ls -1 {shq(parent)} || true'])
        out = cli.api.exec_start(exec_id, stream=False, demux=False)
        s = out.decode(errors="replace") if isinstance(out, (bytes, bytearray)) else str(out)
        return any(line.strip() == base for line in s.splitlines())
    except Exception:
        return False

def _find_entry(container_name: str, study_path: str, entry: str) -> Optional[str]:
    candidates = [
        f"/workspace/{entry}",
        f"/workspace/data/{entry}",
        f"/app/data/{entry}",
        f"/workspace/code/{entry}",
        f"/workspace/codebase/{entry}"
    ]
    for p in candidates:
        if _container_path_exists(container_name, p):
            return p
    return None

def _exec_file(container_name: str, study_path: str, container_path: str, lang: str) -> Dict:
    cli = _require_docker()
    c = cli.containers.get(container_name)
    l = (lang or "").lower()
    
    cmd = []
    if l == "r": cmd = ["Rscript", container_path]
    elif l == "python": cmd = ["python3", container_path]
    elif l == "bash": cmd = ["bash", container_path]
    else:
        return {"ok": False, "exit_code": 2, "stdout": "", "stderr": f"Unsupported lang: {lang}", "artifacts": []}

    exec_id = cli.api.exec_create(c.id, cmd, workdir="/workspace")
    output = cli.api.exec_start(exec_id, stream=False, demux=True, tty=False)
    exit_code = cli.api.exec_inspect(exec_id)["ExitCode"]

    stdout, stderr = output
    stdout = (stdout or b"").decode(errors="replace")
    stderr = (stderr or b"").decode(errors="replace")

    _copy_container_outputs(study_path, container_name)
    arts = _list_local_output_files(study_path)

    return {
        "ok": exit_code == 0,
        "exit_code": exit_code,
        "stdout": stdout,
        "stderr": stderr,
        "artifacts": arts,
    }

def orchestrator_plan(study_path: str, code_mode: str) -> str:
    try:
        spec = _read_spec(study_path)
        plan = plan_from_analysis_info(spec, code_mode)
        out = {
            "plan_id": plan.plan_id,
            "steps": [{"name": s.name, "type": s.type, "lang": s.lang, "entry": s.entry} for s in plan.steps],
        }
        return json.dumps(out)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})

def orchestrator_preview_entry(study_path: str, code_mode: str) -> str:
    try:
        spec = _read_spec(study_path)
        plan = plan_from_analysis_info(spec, code_mode)
        step = next((s for s in plan.steps if s.type == "container"), None)
        if not step or not step.entry:
            return json.dumps({"ok": False, "error": "No container step or entry file specified."})

        found = _find_entry(DEFAULT_CONTAINER_NAME, study_path, step.entry)
        if not found:
            return json.dumps({
                "ok": False,
                "error": f"Entry not found: {step.entry}. Check file paths.",
                "entry": step.entry,
            })

        l = (step.lang or "").lower()
        if l == "r": cmd = ["Rscript", found]
        elif l == "python": cmd = ["python3", found]
        elif l == "bash": cmd = ["bash", found]
        else:
             return json.dumps({"ok": False, "error": f"Unsupported lang: {step.lang}"})
        

        return json.dumps({
            "ok": True,
            "plan_id": plan.plan_id,
            "lang": step.lang,
            "entry": step.entry,
            "resolved_path": found,
            "container_command": cmd,
            "command_pretty": " ".join(cmd),
        })
    except Exception as e:
         return json.dumps({"ok": False, "error": str(e)})

def orchestrator_execute_entry(study_path: str, code_mode: str) -> str:
    try:
        study_dir, _, _, _, _ = _paths(study_path)
        out_path = study_dir / "execution_result.json"

        spec = _read_spec(study_path)
        plan = plan_from_analysis_info(spec, code_mode)

        results: Dict[str, Any] = {"plan_id": plan.plan_id, "steps": []}
        results["steps"].append({"name": "prepare-env", "ok": True})
        

        container_steps = [s for s in plan.steps if s.type == "container"]

        if not container_steps:
            return json.dumps({"ok": False, "error": "No entry files found to execute"})
        
        def _check_long_std(text: str, model_name="gpt-4o"):
            enc = tiktoken.encoding_for_model(model_name if model_name else "gpt-4")
            MAX_TOKENS = 20000
            tokens = enc.encode(text)
            
            if len(tokens) <= MAX_TOKENS:
                return text
            else:
                warning_message = """
                ----------- WARNING MESSAGE ------------
                Your code produces too long stdout and stderr. Below shows the first 20000 tokens of the ouput stream.
                If this output stream does not contain the relevant information for your task, you should try again, rewrite your code so that it only outputs the relevant information.
                ----------- TRUNCATED OUTPUT STREAM ------------
                """
                return f"{warning_message}\n{enc.decode(tokens[:MAX_TOKENS])}"

        for step in container_steps:
            if not step.entry:
                continue

            found = _find_entry(DEFAULT_CONTAINER_NAME, study_path, step.entry)
            if not found:
                res = {"ok": False, "error": f"Entry not found at runtime: {step.entry}", "entry": step.entry}
                results["steps"].append(res)
                continue  # or break, depending on whether you want to abort on failure

            ran = _exec_file(DEFAULT_CONTAINER_NAME, study_path, found, step.lang)
            
            results["steps"].append({
                "name": step.name,
                "ok": ran.get("ok", False),
                "exit_code": ran.get("exit_code"),
                "stdout": _check_long_std(ran.get("stdout")),
                "stderr": _check_long_std(ran.get("stderr")),
                "artifacts": ran.get("artifacts", []),
                "entry": step.entry,
                "resolved_path": found,
            })

        # # Compute final status: True only if ALL container steps succeeded
        # results["ok"] = all(s.get("ok", False) for s in results["steps"] if s["name"] != "prepare-env")

        # out_path.write_text(json.dumps(results, indent=2))
        # return json.dumps(results)
        
        # We append the step details regardless of success so agent can see stderr
        
        
        # results["steps"].append({
        #     "name": step.name,
        #     "ok": ran.get("ok", False),
        #     "exit_code": ran.get("exit_code"),
        #     "stdout": _check_long_std(ran.get("stdout")),
        #     "stderr": _check_long_std(ran.get("stderr")),
        #     "artifacts": ran.get("artifacts", []),
        #     "entry": step.entry,
        #     "resolved_path": found,
        # })
        results["ok"] = all(s.get("ok", False) for s in results["steps"] if s["name"] != "prepare-env")

        out_path.write_text(json.dumps(results, indent=2))
        return json.dumps(results)
    except Exception as e:
        return json.dumps({"ok": False, "error": str(e)})
