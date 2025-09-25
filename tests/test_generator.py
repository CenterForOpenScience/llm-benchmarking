# tests/test_generator.py
import os
import json
import logging
from types import SimpleNamespace
from pathlib import Path
import pytest

from generator.design import easy as gen_easy


# ---------------------------
# Fake OpenAI client (mirrors extractor tests' pattern)
# ---------------------------
class FakeTextBlock:
    def __init__(self, text):
        self.text = SimpleNamespace(value=text)

class FakeMsg:
    def __init__(self, role, content):
        self.role = role
        self.content = content

class FakeMessagesAPI:
    def __init__(self, reply_text_blocks):
        # reply_text_blocks: list of strings, each becomes a FakeTextBlock
        self._reply_blocks = [FakeTextBlock(t) for t in reply_text_blocks]

    def list(self, thread_id):
        # Return a single assistant message with the blocks
        return SimpleNamespace(data=[FakeMsg("assistant", self._reply_blocks)])

class FakeRunsAPI:
    def __init__(self, status="completed"):
        self._status = status

    def retrieve(self, thread_id, run_id):
        return SimpleNamespace(status=self._status)

class FakeThreadsAPI:
    def __init__(self, reply_text_blocks, run_status="completed"):
        self.messages = FakeMessagesAPI(reply_text_blocks)
        self.runs = FakeRunsAPI(status=run_status)

    def create_and_run(self, assistant_id, thread):
        # Return ids used later by retrieve/list
        return SimpleNamespace(thread_id="th_1", id="run_1")

class FakeAssistantsAPI:
    def create(self, name, instructions, model, tools):
        return SimpleNamespace(id="asst_1")

class FakeClient:
    """Provides .beta.assistants and .beta.threads trees."""
    def __init__(self, reply_text_blocks, run_status="completed"):
        self.beta = SimpleNamespace(
            assistants=FakeAssistantsAPI(),
            threads=FakeThreadsAPI(reply_text_blocks, run_status=run_status)
        )


# ---------------------------
# Small logger helper
# ---------------------------
@pytest.fixture
def test_logger():
    logger = logging.getLogger("gen-easy-tests")
    logger.setLevel(logging.DEBUG)
    # keep handler-less to avoid noisy output; Python's default warns only once
    return logger


# ---------------------------
# Utilities: temp study inputs for load_inputs() / build_message_context()
# ---------------------------
def write_json(p: Path, obj):
    p.write_text(json.dumps(obj), encoding="utf-8")

def write_text(p: Path, s: str):
    p.write_text(s, encoding="utf-8")


# ---------------------------
# Unit tests start here
# ---------------------------

def test_load_inputs_reads_expected_files(tmp_path):
    study = tmp_path

    # Files expected by load_inputs()
    write_json(study / "post_registration.json", {"k": 1})
    write_json(study / "replication_info.json", {"r": True})
    # Note: file name follows module's spelling "info_exractor_validation_results.json"
    write_json(study / "info_exractor_validation_results.json", {"valid": "ok"})
    (study / "inputs").mkdir()
    write_text(study / "inputs" / "initial_details_easy.txt", "easy details")
    write_text(study / "inputs" / "initial_details_medium_hard.txt", "mh details")
    # optional PDF existence check
    write_text(study / "inputs" / "original_paper.pdf", "pdf bytes?")

    out = gen_easy.load_inputs(str(study))
    assert out["post_registration"] == {"k": 1}
    assert out["replication_info"] == {"r": True}
    assert out["validation"] == {"valid": "ok"}
    assert out["initial_easy"] == "easy details"
    assert out["initial_medhard"] == "mh details"
    assert out["original_pdf_present"] is True


def test_build_prompt_design_easy_contains_schema_and_rules():
    schema = {"a": 1, "nested": {"b": [1, 2]}}
    prompt = gen_easy.build_prompt_design_easy(schema)
    # Contains the "EXPECTED OUTPUT SHAPE" section & schema keys
    assert "EXPECTED OUTPUT SHAPE:" in prompt
    assert '"a": 1' in prompt
    assert '"nested"' in prompt
    # A couple of rule markers to ensure core instructions are present
    assert "STRICTLY output valid JSON" in prompt
    assert "Do not add extra keys." in prompt


def test_strip_code_fences_variants():
    raw = '{"ok": true}'
    fenced = "```json\n" + raw + "\n```"
    fenced_no_lang = "```\n" + raw + "\n```"

    assert gen_easy.strip_code_fences(raw) == raw
    assert gen_easy.strip_code_fences(fenced) == raw
    assert gen_easy.strip_code_fences(fenced_no_lang) == raw


def test_build_message_context_includes_expected_keys(tmp_path):
    # Simulate directory contents + inputs dict
    (tmp_path / "foo.txt").write_text("x")
    inputs = {
        "post_registration": {"P": 1},
        "replication_info": {"R": 2},
        "validation": {"V": 3},
        "initial_easy": "EASY",
        "initial_medhard": "MH"
    }
    msg = gen_easy.build_message_context(str(tmp_path), inputs)
    data = json.loads(msg)
    assert data["study_path"] == str(tmp_path)
    assert "files_present" in data and "foo.txt" in data["files_present"]
    assert data["post_registration"] == {"P": 1}
    assert data["replication_info"] == {"R": 2}
    assert data["info_extractor_validation_results"] == {"V": 3}
    assert data["initial_details_easy.txt"] == "EASY"
    assert data["initial_details_medium_hard.txt"] == "MH"


def test_run_assistant_parses_json_success(monkeypatch, test_logger):
    # set fake client returning valid JSON (with and without fences)
    reply = '{"x": 1, "ok": true}'
    monkeypatch.setattr(gen_easy, "client", FakeClient([reply]), raising=False)
    out = gen_easy.run_assistant("PROMPT", "USER MSG", test_logger)
    assert out == {"x": 1, "ok": True}

    # fenced JSON
    fenced = "```json\n" + reply + "\n```"
    monkeypatch.setattr(gen_easy, "client", FakeClient([fenced]), raising=False)
    out2 = gen_easy.run_assistant("PROMPT", "USER MSG", test_logger)
    assert out2 == {"x": 1, "ok": True}


def test_run_assistant_handles_bad_json(monkeypatch, test_logger):
    monkeypatch.setattr(gen_easy, "client", FakeClient(['{not json!!!}']), raising=False)
    out = gen_easy.run_assistant("PROMPT", "USER MSG", test_logger)
    assert out is None


def test_run_assistant_handles_failed_status(monkeypatch, test_logger):
    # status != "completed" should return None
    monkeypatch.setattr(gen_easy, "client", FakeClient(['{"a":1}'], run_status="failed"), raising=False)
    out = gen_easy.run_assistant("PROMPT", "USER MSG", test_logger)
    assert out is None


def test_run_design_easy_llm_path(monkeypatch, tmp_path, test_logger):
    # Arrange: ensure OPENAI key is considered present and a fake client is used
    # The module caches OPENAI_API_KEY at import; patch the attribute directly.
    monkeypatch.setattr(gen_easy, "OPENAI_API_KEY", "sk-test", raising=False)

    # Build combined schema (spy value)
    schema = {"schema_key": "val"}
    monkeypatch.setattr(gen_easy, "build_combined_template", lambda td: schema, raising=False)

    # Stub inputs loader (avoid file I/O complexity)
    dummy_inputs = {
        "post_registration": {"orig": 1},
        "replication_info": {"rep": 2},
        "validation": {},
        "initial_easy": "E",
        "initial_medhard": "MH",
        "original_pdf_present": False,
    }
    monkeypatch.setattr(gen_easy, "load_inputs", lambda sp: dummy_inputs, raising=False)

    # Spy for save_prompt_log and write_json
    calls = {"save_prompt_log": [], "write_json": []}
    def save_prompt_log_spy(study_path, stage, prompt, msg):
        calls["save_prompt_log"].append((study_path, stage, prompt, msg))
    def write_json_spy(path, obj):
        calls["write_json"].append((path, obj))
    monkeypatch.setattr(gen_easy, "save_prompt_log", save_prompt_log_spy, raising=False)
    monkeypatch.setattr(gen_easy, "write_json", write_json_spy, raising=False)

    # Force run_assistant to produce a deterministic prereg
    expected = {"prereg": True, "tier": "easy"}
    monkeypatch.setattr(gen_easy, "run_assistant", lambda p, m, l: expected, raising=False)

    # Act
    out = gen_easy.run_design_easy(str(tmp_path), templates_dir=str(tmp_path / "tpls"), show_prompt=False, logger=test_logger)

    # Assert
    assert out == expected
    # Check prompt log called and output path correct
    assert calls["save_prompt_log"], "save_prompt_log should be called"
    assert calls["write_json"], "write_json should be called"
    out_path, out_obj = calls["write_json"][0]
    assert out_path.endswith("preregistration_design.json")
    assert out_obj == expected


def test_run_design_easy_no_key_triggers_fallback(monkeypatch, tmp_path, test_logger):
    # Simulate missing key
    monkeypatch.setattr(gen_easy, "OPENAI_API_KEY", "", raising=False)

    # Combined schema & inputs
    schema = {"K": 1}
    monkeypatch.setattr(gen_easy, "build_combined_template", lambda td: schema, raising=False)

    dummy_inputs = {
        "post_registration": {},
        "replication_info": {},
        "validation": {},
        "initial_easy": "",
        "initial_medhard": "",
        "original_pdf_present": False,
    }
    monkeypatch.setattr(gen_easy, "load_inputs", lambda sp: dummy_inputs, raising=False)

    calls = {"write_json": [], "save_prompt_log": []}
    monkeypatch.setattr(gen_easy, "save_prompt_log", lambda *a, **k: calls["save_prompt_log"].append(a), raising=False)

    # fallback builder spy
    fallback_val = {"built": "fallback"}
    monkeypatch.setattr(gen_easy, "fallback_build", lambda schema, inputs: fallback_val, raising=False)

    monkeypatch.setattr(gen_easy, "write_json", lambda path, obj: calls["write_json"].append((path, obj)), raising=False)

    # run_assistant should not be called; if it is, fail
    monkeypatch.setattr(gen_easy, "run_assistant", lambda *a, **k: (_ for _ in ()).throw(AssertionError("run_assistant should not be called when no API key")), raising=False)

    out = gen_easy.run_design_easy(str(tmp_path), templates_dir=str(tmp_path / "tpls"), show_prompt=True, logger=test_logger)
    assert out == fallback_val
    assert calls["save_prompt_log"]  # still logs prompt/message
    assert calls["write_json"][0][1] == fallback_val


def test_run_design_easy_llm_none_then_fallback(monkeypatch, tmp_path, test_logger):
    # Key present
    monkeypatch.setattr(gen_easy, "OPENAI_API_KEY", "sk-test", raising=False)

    schema = {"S": 1}
    monkeypatch.setattr(gen_easy, "build_combined_template", lambda td: schema, raising=False)

    inputs = {
        "post_registration": {"a": 1},
        "replication_info": {"b": 2},
        "validation": {"v": 3},
        "initial_easy": "e",
        "initial_medhard": "mh",
        "original_pdf_present": True,
    }
    monkeypatch.setattr(gen_easy, "load_inputs", lambda sp: inputs, raising=False)

    # LLM returns None → fallback kicks in
    monkeypatch.setattr(gen_easy, "run_assistant", lambda p, m, l: None, raising=False)

    fallback_val = {"built": "fallback-2"}
    monkeypatch.setattr(gen_easy, "fallback_build", lambda schema, inputs: fallback_val, raising=False)

    captured = []
    monkeypatch.setattr(gen_easy, "write_json", lambda path, obj: captured.append((path, obj)), raising=False)
    monkeypatch.setattr(gen_easy, "save_prompt_log", lambda *a, **k: None, raising=False)

    out = gen_easy.run_design_easy(str(tmp_path), templates_dir="tpls", show_prompt=False, logger=test_logger)
    assert out == fallback_val
    assert captured and captured[0][1] == fallback_val

