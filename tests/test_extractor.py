# tests/test_extractor.py
import json
from types import SimpleNamespace
import pathlib
import builtins
import pytest

import info_extractor.extractor as extractor

class FakeTextBlock:
    """Mimics openai.types.beta.threads.TextContentBlock enough for isinstance() checks."""
    def __init__(self, text):
        # extractor expects .text.value
        self.text = SimpleNamespace(value=text)

class FakeMsg:
    def __init__(self, role, content):
        self.role = role
        self.content = content

class FakeMessagesAPI:
    def __init__(self, reply_text):
        self._reply_text = reply_text

    def list(self, thread_id):
        # return one assistant message containing a text block
        return SimpleNamespace(data=[FakeMsg("assistant", [FakeTextBlock(self._reply_text)])])

class FakeRunsAPI:
    def __init__(self, status="completed"):
        self._status = status

    def retrieve(self, thread_id, run_id):
        return SimpleNamespace(status=self._status)

class FakeThreadsAPI:
    def __init__(self, reply_text):
        self.messages = FakeMessagesAPI(reply_text)
        self.runs = FakeRunsAPI()

    def create_and_run(self, assistant_id, thread):
        # return ids that extractor then uses in retrieve/list
        return SimpleNamespace(thread_id="th_1", id="run_1")

class FakeAssistantsAPI:
    def create(self, name, instructions, model, tools):
        return SimpleNamespace(id="asst_1")

class FakeClient:
    """Provides .beta.assistants and .beta.threads trees."""
    def __init__(self, reply_text):
        self.beta = SimpleNamespace(assistants=FakeAssistantsAPI(),threads=FakeThreadsAPI(reply_text))

# fixtures that wire up the extractor module
@pytest.fixture
def templates(tmp_path, monkeypatch):
    """Create minimal instruction/template files and bind TEMPLATE_PATHS."""
    post_reg = tmp_path / "post_registration_template.json"
    pre_reg = tmp_path / "pre_registration_template.json"
    instructions = tmp_path / "info_extractor_instructions.json"

    post_reg.write_text(json.dumps({"template": "post"}))
    pre_reg.write_text(json.dumps({"template": "pre"}))
    instructions.write_text(json.dumps({
        "easy": {
            "stage_1": {"rule": "s1"},
            "stage_2": {"rule": "s2"}
        }
    }))

    monkeypatch.setattr(extractor, "TEMPLATE_PATHS", {
        "post_registration_template": str(post_reg),
        "pre_registration_template": str(pre_reg),
        "info_extractor_instructions": str(instructions),
    }, raising=False)

    # not used directly, but extractor imports it
    monkeypatch.setattr(extractor, "FILE_SELECTION_RULES", {}, raising=False)
    monkeypatch.setattr(extractor, "API_KEY", "sk-test", raising=False)

@pytest.fixture
def patch_builders(monkeypatch):
    """Mock out builder & io helpers and capture calls."""
    # simple deterministic builders
    monkeypatch.setattr(extractor, "build_prompt", lambda *a, **k: "PROMPT", raising=False)
    monkeypatch.setattr(
        extractor,
        "build_context_and_message",
        lambda study_path, template, file_context, stage, **kw: ("CTX", f"MSG for {stage}"),
        raising=False
    )

    # make read_file_contents return predictable tuple
    def fake_read_file_contents(study_path, difficulty, rules, stage):
        file_ctx = {"some": "context"}
        datasets_original = []
        datasets_replication = []
        code_file_descriptions = []
        original_study_data = {"original_key": "original_val"}
        return file_ctx, datasets_original, datasets_replication, code_file_descriptions, original_study_data

    monkeypatch.setattr(extractor, "read_file_contents", fake_read_file_contents, raising=False)

    # spies for save_* to assert what was saved
    saved = {"calls": []}

    def save_output_spy(extracted_json, study_path, stage):
        saved["calls"].append(("save_output", stage, study_path, extracted_json))

    def save_prompt_log_spy(study_path, stage, prompt, full_message):
        saved["calls"].append(("save_prompt_log", stage, study_path, prompt, full_message))

    monkeypatch.setattr(extractor, "save_output", save_output_spy, raising=False)
    monkeypatch.setattr(extractor, "save_prompt_log", save_prompt_log_spy, raising=False)

    return saved

@pytest.fixture
def swap_text_block(monkeypatch):
    """Ensure isinstance(block, TextContentBlock) works using our FakeTextBlock."""
    monkeypatch.setattr(extractor, "TextContentBlock", FakeTextBlock, raising=False)

# tests 
def test_stage1_parses_markdown_json_and_saves(monkeypatch, templates, patch_builders, swap_text_block, tmp_path):
    # Reply wrapped in ```json fences
    reply_text = """```json
    {"a":1,"stage":"one"}
    ```"""
    # inject fake client
    monkeypatch.setattr(extractor, "client", FakeClient(reply_text), raising=False)

    result = extractor.run_stage_1(study_path=str(tmp_path), difficulty="easy", show_prompt=True)

    assert result == {"a": 1, "stage": "one"}

    # check that prompt log & output were saved
    calls = patch_builders["calls"]
    assert any(c[0] == "save_prompt_log" and c[1] == "stage_1" for c in calls)
    so = [c for c in calls if c[0] == "save_output" and c[1] == "stage_1"]
    assert so, "save_output(stage_1) should be called"
    assert so[0][3] == {"a": 1, "stage": "one"}

def test_stage2_merges_with_original_and_saves(monkeypatch, templates, patch_builders, swap_text_block, tmp_path):
    # Replication part includes a key that should overwrite/augment original
    reply_text = json.dumps({
        "original_study": {"should_be_ignored": True},
        "replication_key": 42
    })
    monkeypatch.setattr(extractor, "client", FakeClient(reply_text), raising=False)

    result = extractor.run_stage_2(study_path=str(tmp_path), difficulty="easy", show_prompt=False)

    # Should merge original_study_data from read_file_contents() with replication_part (excluding "original_study")
    assert result["original_key"] == "original_val"
    assert result["replication_key"] == 42
    assert "should_be_ignored" not in result

    calls = patch_builders["calls"]
    so = [c for c in calls if c[0] == "save_output" and c[1] == "stage_2"]
    assert so
    assert so[0][3] == result

def test_stage1_plain_json_works(monkeypatch, templates, patch_builders, swap_text_block, tmp_path):
    reply_text = '{"ok": true}'
    monkeypatch.setattr(extractor, "client", FakeClient(reply_text), raising=False)

    result = extractor.run_stage_1(study_path=str(tmp_path), difficulty="easy")
    assert result == {"ok": True}

def test_no_text_content_block_raises(monkeypatch, templates, patch_builders, tmp_path):
    """Extractor expects TextContentBlock; if not present it should raise."""
    # Create a content block that is NOT an instance of extractor.TextContentBlock
    class OtherBlock:
        def __init__(self, text):
            self.text = SimpleNamespace(value=text)

    class MessagesNoTextBlock(FakeMessagesAPI):
        def list(self, thread_id):
            return SimpleNamespace(data=[FakeMsg("assistant", [OtherBlock('{"x":1}')])])

    class ThreadsNoTextBlock(FakeThreadsAPI):
        def __init__(self):
            self.messages = MessagesNoTextBlock(None)
            self.runs = FakeRunsAPI()

    class ClientNoText:
        def __init__(self):
            self.beta = SimpleNamespace(assistants=FakeAssistantsAPI(),threads=ThreadsNoTextBlock())

    monkeypatch.setattr(extractor, "client", ClientNoText(), raising=False)

    with pytest.raises(ValueError, match="No TextContentBlock"):
        extractor.run_stage_1(study_path=str(tmp_path), difficulty="easy")

def test_dispatcher_calls_correct_stage(monkeypatch, templates):
    # monkeypatch stage functions to verify dispatch + return sentinel values
    monkeypatch.setattr(extractor, "run_stage_1", lambda *a, **k: {"which": 1})
    monkeypatch.setattr(extractor, "run_stage_2", lambda *a, **k: {"which": 2})

    assert extractor.run_extraction("p", "easy", "stage_1") == {"which": 1}
    assert extractor.run_extraction("p", "easy", "stage_2") == {"which": 2}
    with pytest.raises(ValueError):
        extractor.run_extraction("p", "easy", "bogus")

