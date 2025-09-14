import os, json
from typing import Any, Dict, Optional

def read_json(path: str) -> Optional[Dict[str, Any]]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        try:
            return json.load(f)
        except json.JSONDecodeError:
            return None

def read_text(path: str) -> Optional[str]:
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()

def write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def save_prompt_log(study_path: str, stage: str, prompt: str, message: str) -> None:
    logs_dir = os.path.join(study_path, "_logs")
    os.makedirs(logs_dir, exist_ok=True)
    with open(os.path.join(logs_dir, f"{stage}_prompt.txt"), "w", encoding="utf-8") as f:
        f.write(prompt)
    with open(os.path.join(logs_dir, f"{stage}_message.txt"), "w", encoding="utf-8") as f:
        f.write(message)


