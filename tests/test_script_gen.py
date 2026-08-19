"""Tests for pipeline/script_gen.py.

Everything here is offline: the Gemini SDK is stubbed and the backoff is
recorded rather than served, so the suite never spends a token or a second.
For a live check run `python pipeline/script_gen.py "..."` by hand.

    python -m pytest tests/test_script_gen.py -v
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
from google import genai  # noqa: E402
from google.genai import errors  # noqa: E402

import config  # noqa: E402
from pipeline import caption, script_gen  # noqa: E402


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeGeminiResponse:
    """Only the attributes _extract_text reaches for."""

    def __init__(self, text="", block_reason=None, finish_reason=None):
        self.text = text
        self.prompt_feedback = type("F", (), {"block_reason": block_reason})()
        candidate = type("C", (), {
            "finish_reason": finish_reason, "finish_message": None,
        })()
        self.candidates = [candidate] if finish_reason or text else []


@pytest.fixture
def scripted_gemini(monkeypatch):
    """Queue outcomes per attempt. An exception instance is raised, not returned."""
    calls = []

    def _install(*outcomes):
        class FakeModels:
            def generate_content(self, **kwargs):
                calls.append(kwargs)
                outcome = outcomes[min(len(calls) - 1, len(outcomes) - 1)]
                if isinstance(outcome, BaseException):
                    raise outcome
                return outcome

        class FakeClient:
            def __init__(self, api_key=None):
                self.models = FakeModels()

        monkeypatch.setattr(genai, "Client", FakeClient)
        return calls

    return _install


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    """Record the backoff instead of serving it."""
    slept = []
    monkeypatch.setattr(script_gen.time, "sleep", slept.append)
    return slept


def api_error(code: int, message: str = "boom"):
    return errors.APIError(code, {"error": {"message": message}})


# ---------------------------------------------------------------------------
# Retries - the one Gemini call three steps share
# ---------------------------------------------------------------------------

def test_a_503_is_retried_and_then_succeeds(scripted_gemini):
    calls = scripted_gemini(
        api_error(503, "model is overloaded"),
        FakeGeminiResponse("The ocean is salty."),
    )
    text = script_gen.call_gemini("prompt", system_instruction="be brief")

    assert text == "The ocean is salty."
    assert len(calls) == 2


@pytest.mark.parametrize("code", sorted(script_gen.RETRY_STATUS_CODES))
def test_every_retryable_code_is_retried(code, scripted_gemini):
    calls = scripted_gemini(api_error(code), FakeGeminiResponse("ok"))
    assert script_gen.call_gemini("p", system_instruction="s") == "ok"
    assert len(calls) == 2


@pytest.mark.parametrize("code", [400, 401, 403, 404])
def test_a_settled_failure_is_not_retried(code, scripted_gemini):
    """A bad key or an unknown model fails the same way every time."""
    calls = scripted_gemini(api_error(code, "nope"))
    with pytest.raises(script_gen.ScriptGenerationError, match="nope"):
        script_gen.call_gemini("p", system_instruction="s")
    assert len(calls) == 1


def test_retries_are_bounded_and_the_real_message_survives(scripted_gemini):
    calls = scripted_gemini(api_error(503, "model is overloaded"))
    with pytest.raises(script_gen.ScriptGenerationError) as exc:
        script_gen.call_gemini("p", system_instruction="s")

    assert len(calls) == script_gen.MAX_ATTEMPTS
    assert "model is overloaded" in str(exc.value)
    assert "503" in str(exc.value)


def test_a_dropped_connection_is_retried(scripted_gemini):
    import httpx

    calls = scripted_gemini(
        httpx.ConnectError("connection refused"), FakeGeminiResponse("ok"),
    )
    assert script_gen.call_gemini("p", system_instruction="s") == "ok"
    assert len(calls) == 2


def test_a_programming_error_is_not_retried(scripted_gemini):
    calls = scripted_gemini(TypeError("unexpected keyword"))
    with pytest.raises(script_gen.ScriptGenerationError, match="TypeError"):
        script_gen.call_gemini("p", system_instruction="s")
    assert len(calls) == 1


def test_a_blocked_prompt_is_not_retried(scripted_gemini):
    """A refusal is settled, not transient - _extract_text raises outside the loop."""
    calls = scripted_gemini(FakeGeminiResponse(block_reason="SAFETY"))
    with pytest.raises(script_gen.ScriptGenerationError, match="blocked"):
        script_gen.call_gemini("p", system_instruction="s")
    assert len(calls) == 1


def test_an_empty_candidate_is_not_retried(scripted_gemini):
    calls = scripted_gemini(FakeGeminiResponse(finish_reason="MAX_TOKENS"))
    with pytest.raises(script_gen.ScriptGenerationError, match="empty script"):
        script_gen.call_gemini("p", system_instruction="s")
    assert len(calls) == 1


def test_backoff_grows_between_attempts(scripted_gemini, _no_real_sleeping):
    scripted_gemini(api_error(503))
    with pytest.raises(script_gen.ScriptGenerationError):
        script_gen.call_gemini("p", system_instruction="s")

    assert _no_real_sleeping == [
        script_gen.RETRY_BASE_SECONDS, script_gen.RETRY_BASE_SECONDS * 2,
    ]


def test_a_success_never_sleeps(scripted_gemini, _no_real_sleeping):
    scripted_gemini(FakeGeminiResponse("ok"))
    script_gen.call_gemini("p", system_instruction="s")
    assert _no_real_sleeping == []


def test_the_retry_asks_for_exactly_the_same_thing(scripted_gemini):
    calls = scripted_gemini(api_error(429), FakeGeminiResponse("ok"))
    script_gen.call_gemini("p", system_instruction="s", model="gemini-test")

    assert calls[0]["contents"] == calls[1]["contents"]
    assert calls[0]["model"] == calls[1]["model"] == "gemini-test"


def test_generate_script_rides_out_a_blip(scripted_gemini):
    scripted_gemini(
        api_error(503), FakeGeminiResponse("Rain is acidic. The salt stays."),
    )
    result = script_gen.generate_script("why the ocean is salty", 30)
    assert result["script"] == "Rain is acidic. The salt stays."


# ---------------------------------------------------------------------------
# Saving - the .txt/.json pair step 3 and step 6 expect
# ---------------------------------------------------------------------------

RESULT = {
    "topic": "Why the Ocean Is Salty!",
    "script": "Rain is acidic.\nThe salt stays.",
    "estimated_duration": 12,
}


@pytest.fixture
def scripts_dir(monkeypatch, tmp_path):
    """Keep the suite out of the real output/scripts/."""
    directory = tmp_path / "scripts"
    monkeypatch.setattr(config, "SCRIPTS_DIR", directory)
    return directory


def test_the_txt_holds_the_words_and_nothing_else(scripts_dir):
    saved = script_gen.save_script(RESULT)
    written = Path(saved["script_path"]).read_text(encoding="utf-8")

    # step 3 speaks this file verbatim - a header would be read aloud
    assert written.strip() == RESULT["script"]


def test_the_sidecar_carries_the_topic(scripts_dir):
    saved = script_gen.save_script(RESULT)
    data = json.loads(Path(saved["metadata_path"]).read_text(encoding="utf-8"))

    assert data["topic"] == RESULT["topic"]
    assert data["estimated_duration"] == RESULT["estimated_duration"]


def test_the_default_name_is_a_slug_of_the_topic(scripts_dir):
    saved = script_gen.save_script(RESULT)
    assert Path(saved["script_path"]).name == "why-the-ocean-is-salty.txt"
    assert Path(saved["metadata_path"]).name == "why-the-ocean-is-salty.json"


def test_a_second_script_on_one_topic_does_not_overwrite_the_first(scripts_dir):
    """Two runs of a topic are two reels - step 4 keys off the script's hash."""
    first = script_gen.save_script(RESULT)
    second = script_gen.save_script({**RESULT, "script": "Different words."})

    assert first["script_path"] != second["script_path"]
    assert Path(second["script_path"]).name == "why-the-ocean-is-salty-2.txt"
    assert (Path(first["script_path"]).read_text(encoding="utf-8").strip()
            == RESULT["script"])


def test_the_pair_is_numbered_together(scripts_dir):
    """A stem is only free when both names are - the pair must not split up."""
    scripts_dir.mkdir(parents=True)
    (scripts_dir / "why-the-ocean-is-salty.json").write_text("{}", encoding="utf-8")

    saved = script_gen.save_script(RESULT)
    assert Path(saved["script_path"]).stem == Path(saved["metadata_path"]).stem
    assert Path(saved["script_path"]).name == "why-the-ocean-is-salty-2.txt"


def test_an_explicit_path_is_honoured(tmp_path):
    out = tmp_path / "nested" / "coffee.txt"
    saved = script_gen.save_script(RESULT, out)

    assert Path(saved["script_path"]) == out.resolve()
    assert (tmp_path / "nested" / "coffee.json").is_file()


@pytest.mark.parametrize("bad", ["out.json", "out.mp3", "out"])
def test_an_explicit_path_must_be_a_txt(bad, tmp_path):
    """Anything else would collide with the sidecar written beside it."""
    with pytest.raises(ValueError, match="must end in .txt"):
        script_gen.save_script(RESULT, tmp_path / bad)


def test_an_unnameable_topic_still_gets_a_file(scripts_dir):
    saved = script_gen.save_script({**RESULT, "topic": "???"})
    assert Path(saved["script_path"]).name == "script.txt"


def test_a_non_ascii_topic_is_transliterated(scripts_dir):
    saved = script_gen.save_script({**RESULT, "topic": "café culture in Paris"})
    assert Path(saved["script_path"]).name == "cafe-culture-in-paris.txt"


def test_step_6_reads_the_topic_back_out_of_the_pair(scripts_dir):
    """The contract caption.py documents: ocean.txt and ocean.json side by side."""
    saved = script_gen.save_script(RESULT)
    source = caption.resolve_source(Path(saved["script_path"]))

    assert source.topic == RESULT["topic"]
    assert source.script == RESULT["script"]
