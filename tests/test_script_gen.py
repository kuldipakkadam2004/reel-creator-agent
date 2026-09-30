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
            def __init__(self, api_key=None, http_options=None):
                self.models = FakeModels()
                self.http_options = http_options

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


def test_every_call_is_given_a_deadline(monkeypatch):
    """A request that is never answered raises nothing, so nothing retries it."""
    monkeypatch.setattr(config, "GEMINI_TIMEOUT_SECONDS", 90.0)
    built = []

    class FakeModels:
        def generate_content(self, **kwargs):
            return FakeGeminiResponse("ok")

    class RecordingClient:
        def __init__(self, api_key=None, http_options=None):
            built.append(http_options)
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", RecordingClient)
    script_gen.call_gemini("p", system_instruction="s")

    assert built[0].timeout == 90_000      # HttpOptions counts in milliseconds


def test_a_call_that_never_answers_is_retried_like_any_other_blip(
        scripted_gemini, _no_real_sleeping):
    """What the deadline buys: a hang becomes an error the retry already knows."""
    import httpx

    calls = scripted_gemini(httpx.ReadTimeout("timed out"),
                            FakeGeminiResponse("ok"))

    assert script_gen.call_gemini("p", system_instruction="s") == "ok"
    assert len(calls) == 2


def test_a_call_the_timeout_does_not_end_is_abandoned_and_retried(
        monkeypatch, _no_real_sleeping):
    """2026-09-30: a step 2 call held its connection for 44 minutes with the
    120s timeout set, and nothing raised. Here the first call never returns at
    all - which no transport timeout can see - and the run still moves on."""
    import threading

    monkeypatch.setattr(config, "GEMINI_TIMEOUT_SECONDS", 0.05)
    release = threading.Event()
    calls, closed = [], []

    class FakeModels:
        def generate_content(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                release.wait(5)          # the hang, until the test lets go
            return FakeGeminiResponse("ok")

    class HangingClient:
        def __init__(self, api_key=None, http_options=None):
            self.models = FakeModels()

        def close(self):
            closed.append(True)

    monkeypatch.setattr(genai, "Client", HangingClient)
    try:
        assert script_gen.call_gemini("p", system_instruction="s") == "ok"
    finally:
        release.set()

    assert len(calls) == 2
    assert closed == [True]              # the stuck client's socket is let go
    assert _no_real_sleeping == [script_gen.RETRY_BASE_SECONDS]


def test_a_deadline_that_never_clears_fails_the_call_instead_of_hanging(
        monkeypatch, _no_real_sleeping):
    import threading

    monkeypatch.setattr(config, "GEMINI_TIMEOUT_SECONDS", 0.02)
    release = threading.Event()

    class FakeModels:
        def generate_content(self, **kwargs):
            release.wait(5)
            return FakeGeminiResponse("too late")

    class HangingClient:
        def __init__(self, api_key=None, http_options=None):
            self.models = FakeModels()

    monkeypatch.setattr(genai, "Client", HangingClient)
    try:
        with pytest.raises(script_gen.ScriptGenerationError,
                           match="GeminiDeadlineError"):
            script_gen.call_gemini("p", system_instruction="s")
    finally:
        release.set()


def test_the_deadline_leaves_a_healthy_call_and_its_errors_alone(monkeypatch):
    """The worker thread must hand back exactly what the SDK did."""
    class Models:
        def __init__(self, outcome):
            self.outcome = outcome

        def generate_content(self, **kwargs):
            if isinstance(self.outcome, BaseException):
                raise self.outcome
            return self.outcome

    class Client:
        def __init__(self, outcome):
            self.models = Models(outcome)

    answer = FakeGeminiResponse("fine")
    assert script_gen.generate_content(Client(answer), model="m") is answer

    refusal = api_error(400, "bad request")
    with pytest.raises(type(refusal)) as caught:
        script_gen.generate_content(Client(refusal), model="m")
    assert caught.value is refusal


def test_generate_script_rides_out_a_blip(scripted_gemini):
    scripted_gemini(
        api_error(503), FakeGeminiResponse("Rain is acidic. The salt stays."),
    )
    result = script_gen.generate_script("why the ocean is salty", 30)
    assert result["script"] == "Rain is acidic. The salt stays."


# ---------------------------------------------------------------------------
# The target length
# ---------------------------------------------------------------------------

def test_no_length_asked_for_takes_the_configured_default(scripted_gemini,
                                                          monkeypatch):
    """It used to be a hardcoded 45 - a number no topic had argued for. Step 0
    now suggests one per topic and passes it in -d; this is the fallback."""
    monkeypatch.setattr(config, "SCRIPT_DEFAULT_SECONDS", 30)
    calls = scripted_gemini(FakeGeminiResponse("Rain is acidic. The salt stays."))
    script_gen.generate_script("why the ocean is salty", hooks=False)
    assert f"{script_gen.words_for_duration(30)} words" in calls[0]["contents"]
    assert "30 seconds long" in calls[0]["contents"]


def test_a_length_that_was_asked_for_wins_over_the_default(scripted_gemini,
                                                           monkeypatch):
    monkeypatch.setattr(config, "SCRIPT_DEFAULT_SECONDS", 30)
    calls = scripted_gemini(FakeGeminiResponse("Rain is acidic. The salt stays."))
    script_gen.generate_script("why the ocean is salty", 45, hooks=False)
    assert "45 seconds long" in calls[0]["contents"]


def test_target_duration_resolves_only_the_absent_case(monkeypatch):
    monkeypatch.setattr(config, "SCRIPT_DEFAULT_SECONDS", 30)
    assert script_gen.target_duration(None) == 30
    assert script_gen.target_duration(45) == 45
    # Not `or`: a caller asking for something falsy is still a caller asking,
    # and 0 has to reach the range check rather than be silently defaulted.
    with pytest.raises(ValueError):
        script_gen.generate_script("salt", 0)


def test_the_cli_falls_back_to_the_configured_default(scripted_gemini,
                                                      monkeypatch, tmp_path,
                                                      capsys):
    monkeypatch.setattr(config, "SCRIPT_DEFAULT_SECONDS", 25)
    scripted_gemini(FakeGeminiResponse("Rain is acidic. The salt stays."))
    assert script_gen.main(["why the ocean is salty", "--no-save",
                            "--no-hook"]) == 0
    assert "TARGET  : 25s" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The model family: what a pro model needs that a flash one does not
# ---------------------------------------------------------------------------

def _config_for(model: str, ceiling: int = script_gen.MAX_OUTPUT_TOKENS):
    return script_gen._build_generation_config(
        model,
        system_instruction="write a reel",
        temperature=script_gen.TEMPERATURE,
        max_output_tokens=ceiling,
        thinking_budget=script_gen.THINKING_BUDGET,
        json_mode=False,
    )


def test_a_pro_model_is_given_room_for_the_thinking_it_cannot_skip():
    """The first pro run of step 2 returned nothing at all.

    A pro model cannot be told to stop thinking, and on 2.5 the reasoning is
    charged against max_output_tokens - so a ceiling sized for a ninety-word
    script was spent before the script started and the call came back empty
    with finish_reason=MAX_TOKENS.
    """
    assert _config_for("gemini-2.5-pro").max_output_tokens == (
        script_gen.MAX_OUTPUT_TOKENS + script_gen.PRO_THINKING_HEADROOM
    )


def test_a_flash_model_keeps_the_ceiling_it_was_given():
    # Flash is told not to think at all, so there is nothing to make room for.
    assert (_config_for("gemini-2.5-flash").max_output_tokens
            == script_gen.MAX_OUTPUT_TOKENS)


def test_the_headroom_is_added_to_whatever_the_caller_asked_for():
    """Step 4 and step 6 have their own opinions about answer length.

    The headroom is added to the caller's ceiling rather than replacing it, so
    a call that wanted a short answer still gets a short answer - it just also
    gets room to reason its way to one.
    """
    assert _config_for("gemini-2.5-pro", 512).max_output_tokens == (
        512 + script_gen.PRO_THINKING_HEADROOM
    )


def test_only_a_flash_model_is_sent_a_zero_thinking_budget():
    # A zero budget is what pro rejects outright, so it is never sent one.
    assert _config_for("gemini-2.5-pro").thinking_config is None
    assert _config_for("gemini-2.5-flash").thinking_config.thinking_budget == 0


def test_the_model_family_is_decided_in_one_place(monkeypatch):
    """Step 4 builds its own generation config and asks the same question."""
    assert script_gen.is_pro_model("gemini-2.5-pro") is True
    assert script_gen.is_pro_model("gemini-2.5-flash") is False

    monkeypatch.setattr(config, "GEMINI_MODEL", "gemini-2.5-pro")
    assert script_gen.is_pro_model() is True


# ---------------------------------------------------------------------------
# The claims: is any of this true?
# ---------------------------------------------------------------------------

GLACIER = (
    "This mountain of ice is moving at your walking speed. It carves whole "
    "valleys as it goes. The ice crystals deform under their own pressure. "
    "This happens slowly. A few centimetres a day."
)


def _verify_gemini(monkeypatch, *, findings, repaired="", recheck=None):
    """Stand in for every call the claim check makes.

    `findings` answers the first check, `repaired` the repair and `recheck`
    the check run over that repair - defaulting to a clean bill.

    Both of the last two take a round-by-round sequence as well as a single
    answer, because the repair may now be bought more than once: `repaired`
    a list of scripts, `recheck` a list of finding-lists. A single answer is
    given to every round, which is what the tests written before the loop
    existed expect. Running past the end of a sequence repeats its last entry.
    """
    calls: list[str] = []
    repairs = [repaired] if isinstance(repaired, str) else list(repaired)
    if recheck is None:
        rechecks: list[list] = []
    elif recheck and isinstance(recheck[0], list):
        rechecks = list(recheck)
    else:
        rechecks = [recheck]

    def nth(seq, i, empty):
        if not seq:
            return empty
        return seq[min(i, len(seq) - 1)]

    def dispatch(prompt, **kwargs):
        calls.append(prompt)
        if "fact-check found these faults" in prompt:
            bought = sum(1 for c in calls
                         if "fact-check found these faults" in c)
            return nth(repairs, bought - 1, "")
        seen = sum(1 for c in calls if "Find every place" in c)
        if seen == 1:
            return json.dumps(findings)
        return json.dumps(nth(rechecks, seen - 2, []))

    monkeypatch.setattr(script_gen, "call_gemini", dispatch)
    return calls


def _repairs_bought(calls: list[str]) -> int:
    return sum(1 for c in calls if "fact-check found these faults" in c)


def test_a_sound_script_costs_one_call_and_changes_nothing(monkeypatch):
    calls = _verify_gemini(monkeypatch, findings=[])
    report = script_gen.verify_claims("glaciers", GLACIER)

    assert len(calls) == 1, "the check, and no repair"
    assert report["findings"] == []
    assert report["blocked"] is False
    assert report["revised"] is False
    assert report["script"] == GLACIER


def test_a_script_that_contradicts_itself_is_repaired(monkeypatch):
    """The shipped glacier reel: walking speed in the opening, a few
    centimetres a day eleven sentences later - a factor of about four
    million, and nothing in the pipeline had ever compared the two."""
    fixed = (
        "This mountain of ice moves a few centimetres a day. It carves whole "
        "valleys as it goes. The ice crystals deform under their own "
        "pressure. This happens slowly. A few centimetres a day."
    )
    calls = _verify_gemini(
        monkeypatch,
        findings=[{
            "sentence": "This mountain of ice is moving at your walking speed.",
            "kind": "contradiction",
            "problem": "the script later says a few centimetres a day",
        }],
        repaired=fixed,
    )
    report = script_gen.verify_claims("glaciers", GLACIER)

    assert len(calls) == 3, "check, repair, re-check"
    assert report["revised"] is True
    assert report["blocked"] is False
    assert report["script"] == fixed


def test_a_repair_that_does_not_clear_the_fault_is_thrown_away(monkeypatch):
    """The incumbent rule the other two passes keep: a rewrite is bought, not
    trusted. This is the pass that must not wave anything through."""
    finding = [{"sentence": "This mountain of ice is moving at your walking "
                            "speed.",
                "kind": "contradiction", "problem": "it says otherwise later"}]
    _verify_gemini(monkeypatch, findings=finding,
                   repaired="This mountain of ice is moving at your walking "
                            "speed. It carves valleys as it goes. It is slow.",
                   recheck=finding)
    report = script_gen.verify_claims("glaciers", GLACIER)

    assert report["revised"] is False
    assert report["blocked"] is True
    assert report["script"] == GLACIER
    assert any("did not clear" in w for w in report["warnings"])


DEEP_SEA = (
    "Sunlight does not go far into the ocean. The water strips out red first, "
    "then green, then blue. The ocean does not block the light. By a thousand "
    "feet, ninety-nine percent of the sun is gone. Total darkness."
)

_HALF_FIXED = (
    "Sunlight does not go far into the ocean. The water strips out red first, "
    "then green, then blue. It filters the light away with depth. By a "
    "thousand feet, ninety-nine percent of the sun is gone. Total darkness."
)

_FULLY_FIXED = (
    "Sunlight does not go far into the ocean. The water strips out red first, "
    "then green, then blue. It filters the light away with depth. By a "
    "thousand feet, ninety-nine percent of the sun is gone. Effectively dark."
)

#: The draft refused on 2026-09-13, shortened to the sentences the rounds
#: argued over. Only the findings matter to the loop - it never reads this -
#: but a repair is scored against the script it was handed, so there has to be
#: one.
JELLYFISH = (
    "For every living thing, the clock only runs one way. But one animal found "
    "an escape clause. It collapses back into a formless blob of its earliest "
    "cells."
)

#: Round 1 of that draft: the contradiction answered, the blob still there.
_JELLY_ROUND_1 = (
    "For almost every living thing, the clock only runs one way. But one "
    "animal found an escape clause. It collapses back into a formless blob of "
    "its earliest cells."
)

#: Round 2: the blob answered too, and the escape clause still there.
_JELLY_ROUND_2 = (
    "For almost every living thing, the clock only runs one way. But one "
    "animal found an escape clause. It reverts to an earlier life stage, a "
    "polyp."
)


def test_a_second_round_lands_the_repair_the_first_one_missed(monkeypatch):
    """The deep-sea reel of 2026-09-02. The first rewrite cleared the
    contradiction and left the overstatement standing, and under a single
    repair that cost the entire topic - as it did for five others the same
    day, every one of them refused on an overstatement."""
    calls = _verify_gemini(
        monkeypatch,
        findings=[
            {"sentence": "The ocean does not block the light.",
             "kind": "contradiction",
             "problem": "the water is described stripping light out above"},
            {"sentence": "Total darkness.",
             "kind": "false",
             "problem": "one percent of the light is left, which is not total"},
        ],
        repaired=[_HALF_FIXED, _FULLY_FIXED],
        recheck=[
            [{"sentence": "Total darkness.", "kind": "false",
              "problem": "one percent of the light is left"}],
            [],
        ],
    )
    report = script_gen.verify_claims("the deep sea", DEEP_SEA)

    assert _repairs_bought(calls) == 2, "the second round is the whole point"
    assert report["blocked"] is False
    assert report["revised"] is True
    assert report["script"] == _FULLY_FIXED
    assert report["remaining"] == []


def test_a_lone_finding_that_holds_level_is_still_tried_again(monkeypatch):
    """The run of 2026-09-03, which lost both its topics to this. A script
    with a *single* blocking finding can never leave "strictly fewer" without
    leaving none, so demanding a strict reduction bought one attempt and no
    more - VERIFY_REPAIR_ROUNDS was inert in exactly the common case. The
    repair is a sampled call: it misses the qualifier once and lands it on the
    next ask."""
    monkeypatch.setattr(config, "VERIFY_REPAIR_ROUNDS", 2)
    standing = [{"sentence": "Total darkness.", "kind": "false",
                 "problem": "one percent of the light is left"}]
    calls = _verify_gemini(monkeypatch, findings=standing,
                           repaired=[_HALF_FIXED, _FULLY_FIXED],
                           recheck=[standing, []])
    report = script_gen.verify_claims("the deep sea", DEEP_SEA)

    assert _repairs_bought(calls) == 2, "holding level is not unrepairable"
    assert report["blocked"] is False
    assert report["revised"] is True
    assert report["script"] == _FULLY_FIXED
    assert report["remaining"] == []


def test_a_rewrite_that_never_lands_spends_its_rounds_and_keeps_the_original(
        monkeypatch):
    """Re-sampling is bounded by the budget, not by hope. When every round
    comes back with the same finding standing, the rounds run out and the
    original ships - a rewrite that only ever held level was never an
    improvement and is not recorded as one."""
    monkeypatch.setattr(config, "VERIFY_REPAIR_ROUNDS", 2)
    standing = [{"sentence": "Total darkness.", "kind": "false",
                 "problem": "one percent of the light is left"}]
    # Both rounds hand back a script that still contains "Total darkness.",
    # so the finding survives the check that a finding quotes a sentence the
    # script really has - a rewrite that dropped the sentence would clear it.
    calls = _verify_gemini(monkeypatch, findings=standing,
                           repaired=[_HALF_FIXED, _HALF_FIXED],
                           recheck=[standing, standing])
    report = script_gen.verify_claims("the deep sea", DEEP_SEA)

    assert _repairs_bought(calls) == 2, "the budget, and not one call more"
    assert report["blocked"] is True
    assert report["revised"] is False
    assert report["script"] == DEEP_SEA
    assert any("did not clear" in w for w in report["warnings"])


def test_a_rewrite_that_makes_things_worse_stops_the_rounds(monkeypatch):
    """The one kind that still ends the loop early. A rewrite that comes back
    with more faults than it went in with is going backwards, and the next
    round would start from that worse script."""
    monkeypatch.setattr(config, "VERIFY_REPAIR_ROUNDS", 2)
    standing = [{"sentence": "Total darkness.", "kind": "false",
                 "problem": "one percent of the light is left"}]
    # Both sentences are ones _HALF_FIXED still contains, so both survive the
    # check that a finding quotes a sentence the script really has.
    worse = standing + [
        {"sentence": "By a thousand feet, ninety-nine percent of the sun "
                     "is gone.",
         "kind": "false",
         "problem": "the figure is depth- and water-dependent"},
    ]
    calls = _verify_gemini(monkeypatch, findings=standing,
                           repaired=[_HALF_FIXED, _FULLY_FIXED],
                           recheck=[worse, []])
    report = script_gen.verify_claims("the deep sea", DEEP_SEA)

    assert _repairs_bought(calls) == 1, "it went backwards, so it stopped"
    assert report["blocked"] is True
    assert report["revised"] is False
    assert report["script"] == DEEP_SEA


def test_a_repair_that_keeps_improving_is_not_cut_off_by_the_budget(
        monkeypatch):
    """The tuna draft of 2026-09-05 went 3 blocking findings -> 2 -> 1, and was
    refused there having improved on every round it was given.

    `VERIFY_REPAIR_ROUNDS` exists to stop a rewrite that is going nowhere. It
    was ending one that was working, a round short of clean, after the run had
    already paid for the whole generation."""
    monkeypatch.setattr(config, "VERIFY_REPAIR_ROUNDS", 2)
    monkeypatch.setattr(config, "VERIFY_REPAIR_MAX_ROUNDS", 5)
    three = [{"sentence": "Total darkness.", "kind": "false",
              "problem": "one percent of the light is left"},
             {"sentence": "The water strips out red first, then green, then "
                          "blue.",
              "kind": "false", "problem": "blue is what is left, not stripped"},
             {"sentence": "The ocean does not block the light.",
              "kind": "contradiction", "problem": "it is stripped out above"}]
    two = [{"sentence": "Total darkness.", "kind": "false",
            "problem": "one percent is left"},
           {"sentence": "The ocean does not block the light.",
            "kind": "contradiction", "problem": "it is stripped out above"}]
    one = [{"sentence": "Total darkness.", "kind": "false",
            "problem": "one percent is left"}]
    calls = _verify_gemini(
        monkeypatch, findings=three,
        repaired=[DEEP_SEA.replace("red first", "the reds first"),
                  _HALF_FIXED, _FULLY_FIXED],
        recheck=[two, one, []],
    )

    report = script_gen.verify_claims("the deep sea", DEEP_SEA)

    assert _repairs_bought(calls) == 3, "the third round is the one that lands"
    assert report["blocked"] is False
    assert report["revised"] is True
    assert report["script"] == _FULLY_FIXED
    assert report["remaining"] == []


def test_the_budget_is_spent_only_on_rounds_that_get_nowhere(monkeypatch):
    """Converging and arriving are still not the same thing.

    Three rounds brought the count down and gave the patience back each time;
    the two after that held level and spent it. What is kept is the best script
    of them - still refused, because something on it is still wrong."""
    monkeypatch.setattr(config, "VERIFY_REPAIR_ROUNDS", 2)
    monkeypatch.setattr(config, "VERIFY_REPAIR_MAX_ROUNDS", 9)
    three = [{"sentence": "Total darkness.", "kind": "false",
              "problem": "one percent of the light is left"},
             {"sentence": "The water strips out red first, then green, then "
                          "blue.",
              "kind": "false", "problem": "blue is what is left, not stripped"},
             {"sentence": "The ocean does not block the light.",
              "kind": "contradiction", "problem": "it is stripped out above"}]
    two = [{"sentence": "Total darkness.", "kind": "false",
            "problem": "one percent is left"},
           {"sentence": "The ocean does not block the light.",
            "kind": "contradiction", "problem": "it is stripped out above"}]
    one = [{"sentence": "Total darkness.", "kind": "false",
            "problem": "one percent is left"}]
    calls = _verify_gemini(
        monkeypatch, findings=three,
        repaired=[DEEP_SEA.replace("red first", "the reds first"), _HALF_FIXED],
        recheck=[two, one],       # and `one` again for every round after
    )

    report = script_gen.verify_claims("the deep sea", DEEP_SEA)

    # 3 -> 2 -> 1 buys three rounds back; the fourth and fifth hold level.
    assert _repairs_bought(calls) == 4
    assert report["revised"] is True
    assert report["blocked"] is True
    assert report["script"] == _HALF_FIXED
    assert len(report["remaining"]) == 1
    assert any("still stand" in w for w in report["warnings"])


def test_a_round_that_answers_its_fault_is_kept_even_at_the_same_count(
        monkeypatch):
    """The jellyfish draft of 2026-09-13, as it ran.

    One blocking finding, and a repair that fixed it and turned up a different
    one - twice. The count never moved, because "fewer than one" is zero and
    zero returns clean, so the loop recorded no best round and reverted to the
    original: the draft with the contradiction round 1 had corrected, plus the
    two sentences rounds 1 and 2 went on to fix. It refused the topic on the
    first of those.
    """
    monkeypatch.setattr(config, "VERIFY_REPAIR_ROUNDS", 2)
    started = [{"sentence": "For every living thing, the clock only runs one "
                            "way.",
                "kind": "contradiction",
                "problem": "the immortal jellyfish is the counter-example"}]
    then = [{"sentence": "It collapses back into a formless blob of its "
                         "earliest cells.",
             "kind": "false", "problem": "it reverts to a polyp"}]
    after = [{"sentence": "But one animal found an escape clause.",
              "kind": "false", "problem": "hydra do it too"}]
    _verify_gemini(monkeypatch, findings=started,
                   repaired=[_JELLY_ROUND_1, _JELLY_ROUND_2],
                   recheck=[then, after])

    report = script_gen.verify_claims("why some animals never seem to age",
                                      JELLYFISH)

    assert report["blocked"] is True, "one finding still stands"
    assert report["revised"] is True
    assert report["script"] == _JELLY_ROUND_2, "the last round that answered"
    assert report["remaining"] == after, (
        "the refusal must name the fault that survived, not the one the "
        "draft started with"
    )
    assert not any("kept the original" in w for w in report["warnings"])


def test_a_rewrite_carrying_its_fault_back_is_not_kept_at_the_same_count(
        monkeypatch):
    """The other side of it, and why the count cannot be the test on its own.

    A repair that returns the sentence it was sent to fix, and has the same
    finding read back to it, has bought nothing - it is level with the draft
    on count and identical to it on the only fault either has.
    """
    monkeypatch.setattr(config, "VERIFY_REPAIR_ROUNDS", 1)
    standing = [{"sentence": "Total darkness.", "kind": "false",
                 "problem": "one percent of the light is left"}]
    # Same fault, worded afresh: the checker is a sampled call and writes its
    # reasoning differently each time it reports the same sentence.
    again = [{"sentence": "Total darkness!", "kind": "false",
              "problem": "a hundredth of the surface light reaches that deep"}]
    _verify_gemini(monkeypatch, findings=standing,
                   repaired=_HALF_FIXED, recheck=again)

    report = script_gen.verify_claims("the deep sea", DEEP_SEA)

    assert report["blocked"] is True
    assert report["script"] == DEEP_SEA, "the original still stands"
    assert any("did not clear" in w for w in report["warnings"])


def test_the_ceiling_stops_a_repair_that_is_still_improving(monkeypatch):
    """Strict improvement is self-limiting, but a draft with a dozen faults
    would grind them off one round at a time and bill for every one. At that
    point the draft is the problem and the next topic is cheaper."""
    monkeypatch.setattr(config, "VERIFY_REPAIR_ROUNDS", 2)
    monkeypatch.setattr(config, "VERIFY_REPAIR_MAX_ROUNDS", 2)
    three = [{"sentence": "Total darkness.", "kind": "false",
              "problem": "one percent of the light is left"},
             {"sentence": "The water strips out red first, then green, then "
                          "blue.",
              "kind": "false", "problem": "blue is what is left, not stripped"},
             {"sentence": "The ocean does not block the light.",
              "kind": "contradiction", "problem": "it is stripped out above"}]
    two = [{"sentence": "Total darkness.", "kind": "false",
            "problem": "one percent is left"},
           {"sentence": "The ocean does not block the light.",
            "kind": "contradiction", "problem": "it is stripped out above"}]
    one = [{"sentence": "Total darkness.", "kind": "false",
            "problem": "one percent is left"}]
    calls = _verify_gemini(
        monkeypatch, findings=three,
        repaired=[DEEP_SEA.replace("red first", "the reds first"),
                  _HALF_FIXED, _FULLY_FIXED],
        recheck=[two, one, []],
    )

    report = script_gen.verify_claims("the deep sea", DEEP_SEA)

    assert _repairs_bought(calls) == 2, "the ceiling, not the patience"
    assert report["blocked"] is True
    assert report["script"] == _HALF_FIXED
    assert any("ceiling" in w for w in report["warnings"])


def test_every_repair_round_is_recorded_with_what_it_came_back_with(
        monkeypatch):
    """What the last three attempts at this pass were argued without.

    When no round lands, the script left standing is the original - so
    `remaining` describes the original, which is correct and is also why the
    run log has only ever quoted the fault the draft *started* with. The
    rounds in between left no trace anywhere, and every judgement about
    whether this pass refuses too much was made from that one stale line.
    """
    monkeypatch.setattr(config, "VERIFY_REPAIR_ROUNDS", 2)
    standing = [{"sentence": "Total darkness.", "kind": "false",
                 "problem": "one percent of the light is left"}]
    _verify_gemini(monkeypatch, findings=standing,
                   repaired=[_HALF_FIXED, _HALF_FIXED],
                   recheck=[standing, standing])
    report = script_gen.verify_claims("the deep sea", DEEP_SEA)

    assert report["blocked"] is True
    assert report["script"] == DEEP_SEA, "the original is still what stands"
    assert [a["round"] for a in report["attempts"]] == [1, 2]
    assert all(a["script"] == _HALF_FIXED for a in report["attempts"])
    assert all(a["remaining"] == standing for a in report["attempts"])
    assert sum("repair round" in w for w in report["warnings"]) == 2


def test_the_round_that_clears_it_is_recorded_too(monkeypatch):
    """Not only failures. A round that landed is the evidence that the budget
    is worth what it costs, and it is counted in the same place."""
    standing = [{"sentence": "Total darkness.", "kind": "false",
                 "problem": "one percent of the light is left"}]
    _verify_gemini(monkeypatch, findings=standing,
                   repaired=_FULLY_FIXED, recheck=[])
    report = script_gen.verify_claims("the deep sea", DEEP_SEA)

    assert report["blocked"] is False
    assert report["attempts"] == [{"round": 1, "script": _FULLY_FIXED,
                                   "remaining": []}]
    assert not any("repair round" in w for w in report["warnings"]), (
        "a round that worked is not a warning")


def test_a_refusal_carries_the_draft_and_the_report_that_refused_it(
        monkeypatch):
    """The evidence for a refusal is a whole script and every round spent on
    it; the caller used to get one line of message and drop the rest. Both
    ride on the exception now, and `run.py` writes them to
    output/scripts/refused/ before it falls through to the next candidate."""
    draft = ("Sunlight does not go far into the ocean. "
             "By a thousand feet the sun is gone. "
             "Total darkness.")
    # The repair leaves "Total darkness." word for word, so the re-check's
    # finding still quotes a sentence the script really has - a rewrite that
    # touched it would clear the finding by removing what it is about.
    repaired = draft.replace("By a thousand feet the sun is gone.",
                             "By a thousand feet almost all of it is gone.")
    standing = [{"sentence": "Total darkness.", "kind": "false",
                 "problem": "one percent of the light is left"}]

    def dispatch(prompt, **kwargs):
        if "fact-check found these faults" in prompt:
            return repaired
        if "Find every place" in prompt:
            return json.dumps(standing)
        return draft

    monkeypatch.setattr(script_gen, "call_gemini", dispatch)
    with pytest.raises(script_gen.ScriptGenerationError) as caught:
        script_gen.generate_script("the deep sea", 20,
                                   hooks=False, structure=False)

    assert caught.value.script == draft, "the draft that was refused"
    assert caught.value.report["remaining"] == standing
    assert caught.value.report["attempts"], "and what the rounds came back with"


def test_a_failure_that_is_not_a_refusal_carries_no_draft(monkeypatch):
    """`script` is empty for every other cause, so a caller can tell a script
    that was judged and refused from one that never existed."""
    error = script_gen.ScriptGenerationError("the API is down")

    assert error.script == ""
    assert error.report == {}


def test_save_refused_script_keeps_the_words_and_the_rounds(monkeypatch,
                                                            tmp_path):
    """Two files, named the way `save_script` names its own: the draft as it
    stood, and the report that refused it with every round inside it."""
    monkeypatch.setattr(config, "REFUSED_DIR", tmp_path)
    report = {"remaining": [{"sentence": "Total darkness.", "kind": "false",
                             "problem": "one percent of the light is left"}],
              "attempts": [{"round": 1, "script": _HALF_FIXED,
                            "remaining": []}]}

    sidecar = script_gen.save_refused_script("the deep sea", DEEP_SEA, report)

    assert sidecar.suffix == ".json"
    assert sidecar.with_suffix(".txt").read_text(
        encoding="utf-8").strip() == DEEP_SEA
    kept = json.loads(sidecar.read_text(encoding="utf-8"))
    assert kept["topic"] == "the deep sea"
    assert kept["claims"]["attempts"][0]["script"] == _HALF_FIXED
    assert kept["claims"]["remaining"][0]["sentence"] == "Total darkness."


def test_the_repair_is_told_to_qualify_an_overstatement():
    """The rule the six topics discarded on 2026-09-02 all needed. The repair
    used to be told only that it must not go vague, which forbids the one
    correction an overstatement actually wants: "all that water flashes to
    steam" needs "some of it", and that is more accurate, not vaguer."""
    template = script_gen._VERIFY_FIX_TEMPLATE
    assert "overstated" in template
    assert "Almost all" in template
    assert "go vague in order to survive the check" in template,         "and dodging a false claim is still forbidden"


def test_an_unrepairable_script_blocks(monkeypatch):
    """The Oakland Zoo reel. There is no true version of a crimson mountain
    lion, so the repair cannot land and the topic is a dead end."""
    lion = ("This mountain lion looks like it walked out of a painting. Now "
            "imagine one that is crimson red. Her name is P Twenty Two. She "
            "is now safe at the Oakland Zoo.")
    finding = [{"sentence": "Now imagine one that is crimson red.",
                "kind": "false",
                "problem": "there is no such animal as a crimson mountain lion"}]
    _verify_gemini(monkeypatch, findings=finding, repaired="", recheck=finding)
    report = script_gen.verify_claims("a crimson mountain lion", lion)

    assert report["blocked"] is True
    assert report["remaining"] == finding


def test_a_finding_about_a_sentence_the_script_does_not_have_is_ignored(monkeypatch):
    """A finding that quotes something the script does not contain is a
    finding about a script the model wrote itself, and acting on it means
    repairing a fault that was never there."""
    calls = _verify_gemini(monkeypatch, findings=[{
        "sentence": "Glaciers are made entirely of compressed sardines.",
        "kind": "false", "problem": "they are not",
    }])
    report = script_gen.verify_claims("glaciers", GLACIER)

    assert len(calls) == 1, "nothing was worth repairing"
    assert report["findings"] == []
    assert report["blocked"] is False
    assert any("not in the script" in w for w in report["warnings"])


def test_a_quote_that_lost_its_punctuation_still_matches():
    assert script_gen._finding_is_about(
        GLACIER, "This mountain of ice is moving at your walking speed"
    )


def test_something_merely_unverifiable_does_not_block(monkeypatch):
    """`unsupported` is the model saying it cannot confirm a claim, which is
    true of most interesting ones. Blocking on it ships nothing."""
    calls = _verify_gemini(monkeypatch, findings=[{
        "sentence": "This happens slowly. A few centimetres a day.",
        "kind": "unsupported", "problem": "no source given",
    }])
    report = script_gen.verify_claims("glaciers", GLACIER)

    assert len(calls) == 1, "noted, not repaired"
    assert report["blocked"] is False
    assert report["findings"], "it is still reported"


#: The hook of the reel step 8 lost on 2026-09-06, kept verbatim in
#: output/scripts/refused/. Its first sentence is a metaphor - a tree is not a
#: pump - and every wording of it the repair tried was refused on the mechanics
#: of the image rather than on anything the script was claiming.
TREE = (
    "This tree is a silent water pump with absolutely no moving parts. "
    "It can lift four hundred liters of water a day, ten stories high. "
    "The engine isn't at the bottom. It's at the top."
)


def test_a_metaphor_is_not_a_false_claim(monkeypatch):
    """A hook is compressed on purpose, and read literally a compression is
    always false of something. Blocking on that loses the topic and cannot be
    repaired, because the sentence was never the fault."""
    calls = _verify_gemini(monkeypatch, findings=[{
        "sentence": "This tree is a silent water pump with absolutely no "
                    "moving parts.",
        "kind": "figurative",
        "problem": "Stomata open and close; those are moving parts.",
    }])
    report = script_gen.verify_claims("trees", TREE)

    assert len(calls) == 1, "noted, not repaired"
    assert report["blocked"] is False
    assert report["revised"] is False
    assert report["findings"], "it is still reported"


def test_an_invented_animal_still_blocks_beside_a_metaphor(monkeypatch):
    """The carve-out is for images, not for cover. A script that is wrong
    about something is still refused, and the repair is handed the fault that
    blocks and not the one that does not."""
    calls = _verify_gemini(monkeypatch, findings=[
        {"sentence": "This tree is a silent water pump with absolutely no "
                     "moving parts.",
         "kind": "figurative", "problem": "stomata move"},
        {"sentence": "It can lift four hundred liters of water a day, ten "
                     "stories high.",
         "kind": "false", "problem": "wrong by an order of magnitude"},
    ])
    report = script_gen.verify_claims("trees", TREE)

    assert report["blocked"] is True, "the false claim still costs the topic"
    faults = [c for c in calls if "fact-check found these faults" in c]
    assert faults, "the repair was bought"
    assert "order of magnitude" in faults[0]
    assert "stomata move" not in faults[0], "the image is not sent for repair"


def test_the_checker_is_offered_the_kind_it_is_meant_to_use(monkeypatch):
    """A kind the model is never told about is a kind it never returns, and
    the pass falls straight back to calling every metaphor false."""
    assert "figurative" in script_gen._VERIFY_PROMPT_TEMPLATE
    assert "figurative" in script_gen._VERIFY_KINDS
    assert "figurative" not in script_gen._VERIFY_BLOCKING


def test_the_claim_check_can_be_switched_off_without_calling_anything(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("the model must not be called with VERIFY_MODE=off")

    monkeypatch.setattr(script_gen, "call_gemini", explode)
    report = script_gen.verify_claims("glaciers", GLACIER, mode="off")

    assert report["mode"] == "off"
    assert report["blocked"] is False
    assert report["script"] == GLACIER


def test_a_repair_nobody_could_re_check_is_not_recorded_as_cleared(monkeypatch):
    """The shipwreck reel's script, and why it was the one that shipped.

    An unreachable checker comes back with no findings, and no findings read
    exactly like a clean bill: the repair cleared `len(still) < len(blocking)`
    against an empty list and went out as verified, on the evening three fully
    checked scripts were turned away for faults this one was never tested for.
    """
    fixed = (
        "This mountain of ice moves a few centimetres a day. It carves whole "
        "valleys as it goes. The ice crystals deform under their own "
        "pressure. This happens slowly. A few centimetres a day."
    )
    finding = [{"sentence": "This mountain of ice is moving at your walking "
                            "speed.",
                "kind": "contradiction", "problem": "it says otherwise later"}]
    calls: list[str] = []

    def dispatch(prompt, **kwargs):
        calls.append(prompt)
        if "fact-check found these faults" in prompt:
            return fixed
        if sum(1 for c in calls if "Find every place" in c) == 1:
            return json.dumps(finding)
        raise script_gen.ScriptGenerationError("503 unavailable")

    monkeypatch.setattr(script_gen, "call_gemini", dispatch)
    report = script_gen.verify_claims("glaciers", GLACIER)

    # Still not a reason for the reel not to exist, and still the better of the
    # two scripts on offer - but not something to call checked.
    assert report["blocked"] is False
    assert report["script"] == fixed
    assert report["checked"] is False
    assert any("re-checked" in w for w in report["warnings"])


def test_a_script_that_passes_is_recorded_as_actually_checked(monkeypatch):
    """The flag has to separate the two states, or it says nothing."""
    _verify_gemini(monkeypatch, findings=[])
    report = script_gen.verify_claims("glaciers", GLACIER)

    assert report["checked"] is True


def test_a_checker_that_cannot_be_reached_never_blocks_a_reel(monkeypatch):
    def fail(*args, **kwargs):
        raise script_gen.ScriptGenerationError("503 unavailable")

    monkeypatch.setattr(script_gen, "call_gemini", fail)
    report = script_gen.verify_claims("glaciers", GLACIER)

    assert report["blocked"] is False
    assert report["script"] == GLACIER
    assert any("503" in w for w in report["warnings"])


def test_unparseable_findings_are_survived(monkeypatch):
    monkeypatch.setattr(script_gen, "call_gemini", lambda *a, **k: "not json")
    report = script_gen.verify_claims("glaciers", GLACIER)

    assert report["blocked"] is False
    assert report["warnings"]


def test_a_blocked_script_ends_the_topic_rather_than_shipping(monkeypatch):
    """ScriptGenerationError is already in step 8's _TOPIC_FAILURES, so a
    refused script falls through to the next candidate on the board."""
    finding = [{"sentence": "Now imagine one that is crimson red.",
                "kind": "false", "problem": "no such animal"}]

    def dispatch(prompt, **kwargs):
        if "Write a voiceover narration" in prompt:
            return ("A mountain lion walks out of a painting. Now imagine one "
                    "that is crimson red. Her name is P Twenty Two.")
        if "fact-check found these faults" in prompt:
            return ""
        return json.dumps(finding)

    monkeypatch.setattr(script_gen, "call_gemini", dispatch)
    with pytest.raises(script_gen.ScriptGenerationError) as exc:
        script_gen.generate_script("a crimson mountain lion", 20,
                                   hooks=False, structure=False)

    assert "fact-check" in str(exc.value)
    assert "crimson" in str(exc.value)


def test_a_sound_script_carries_its_claim_report(monkeypatch):
    def dispatch(prompt, **kwargs):
        if "Write a voiceover narration" in prompt:
            return "Salt gets into the sea somehow. Rain carries it there."
        return "[]"

    monkeypatch.setattr(script_gen, "call_gemini", dispatch)
    result = script_gen.generate_script("salt", 20, hooks=False,
                                        structure=False)

    assert result["claims"]["mode"] == "llm"
    assert result["claims"]["blocked"] is False
    assert result["claims"]["findings"] == []


def test_only_findings_the_prompt_asked_for_are_kept():
    parsed = script_gen._parse_verify_json(json.dumps([
        {"sentence": "A real one.", "kind": "false", "problem": "wrong"},
        {"sentence": "Another.", "kind": "vibes", "problem": "off"},
        {"sentence": "", "kind": "false", "problem": "nothing quoted"},
    ]))
    assert [f["sentence"] for f in parsed] == ["A real one."]


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


# ---------------------------------------------------------------------------
# The hook: scored here, not by the model
# ---------------------------------------------------------------------------

def test_throat_clearing_is_punished_hard():
    warm_up = script_gen.score_hook("Have you ever wondered why the ocean is salty?")
    direct = script_gen.score_hook("Rain is the reason your ocean tastes of salt.")

    assert "throat-clearing" in warm_up["components"]
    assert warm_up["score"] < direct["score"]


def test_every_listed_opener_is_actually_caught():
    for opener in script_gen._HOOK_THROAT_CLEARING:
        scored = script_gen.score_hook(f"{opener} something happens here.")
        assert "throat-clearing" in scored["components"], opener


def test_a_concrete_number_is_worth_something():
    with_number = script_gen.score_hook("The ocean gains 4 billion tons of salt a year.")
    without = script_gen.score_hook("The ocean gains a lot of salt a year.")

    assert with_number["components"].get("a number") == 0.8
    assert with_number["score"] > without["score"]


def test_a_number_counts_whether_it_is_typed_in_digits_or_words():
    """The same fact, typed two ways, has to score the same.

    Measured on the deep-ocean run: "heard ten thousand kilometres away" came
    back 0.0 and three of five candidates scored a flat zero, so the pass kept
    an incumbent it had a live challenger for.
    """
    spelled = script_gen.score_hook(
        "A sound in the deep ocean can be heard ten thousand kilometres away."
    )
    digits = script_gen.score_hook(
        "A sound in the deep ocean can be heard 10,000 kilometres away."
    )
    assert spelled["components"].get("a number") == 0.8
    assert spelled["score"] == pytest.approx(digits["score"])


def test_a_hyphenated_number_word_counts_too():
    """The compound shape the structure pass already learned to read."""
    assert script_gen.says_a_number("Ninety-five percent of it is unexplored.")
    assert script_gen.score_hook(
        "Ninety-five percent of the ocean is unexplored."
    )["components"].get("a number") == 0.8


def test_one_as_a_pronoun_is_not_a_number():
    """"No one" and "the one thing" are most of English, not a figure."""
    for line in ("No one knows why the deep ocean sounds like this.",
                 "This is the one thing nobody tells you about the ocean."):
        assert not script_gen.says_a_number(line), line
        assert "a number" not in script_gen.score_hook(line)["components"]


def test_a_bare_magnitude_is_not_a_figure():
    """"Thousands of kilometres" is a hand-wave; the scorer wants a quantity."""
    assert not script_gen.says_a_number(
        "Sound travels thousands of kilometres down there."
    )


def test_a_small_number_word_that_is_only_counting_is_not_a_figure():
    """English uses "two" to enumerate at least as often as to measure.

    Both of these fired the component on the reels as shipped. The first won
    its script on that 0.8 alone, over a candidate carrying second person and
    a strong opener - and it was itself restating the body, a -1.5 the bonus
    was covering for.
    """
    for line in ("An autumn leaf has two kinds of colour: the old and the new.",
                 "Chernobyl is an invoice that two countries will be paying for."):
        assert not script_gen.says_a_number(line), line
        assert "a number" not in script_gen.score_hook(line)["components"]


@pytest.mark.parametrize("line", [
    # a unit of measure
    "These seeds can wait fifty years for a rain that lasts three days.",
    "Almost every great desert on Earth lies at thirty degrees of latitude.",
    "There is a traffic jam that covers seventy percent of the planet.",
    # one figure written in two tokens, only the first of them small
    "Every July, Earth runs five million kilometers farther from the sun.",
    "Bacteria eat two hundred pounds of the Titanic's steel every day.",
    "One kilogram of rust can deoxygenate ten thousand litres of seawater.",
    # a magnitude standing on its own
    "What you are standing on costs a thousand years for a single inch.",
])
def test_a_small_number_word_quantifying_something_still_counts(line):
    """The 27 of 30 the rule must not touch - see _HOOK_MEASURE_WORDS."""
    assert script_gen.says_a_number(line), line


def test_the_enumeration_bonus_no_longer_outranks_a_written_hook():
    """The leaves reel, end to end: the shipped winner loses to the better line."""
    body = ("An autumn leaf has two kinds of colour: the old and the new. "
            "All summer long, the leaf is a factory flooded with green "
            "chlorophyll, hiding the colours sleeping just underneath.")
    shipped = "An autumn leaf has two kinds of colour: the old and the new."
    better = "This leaf was yellow all summer; you just couldn't see it."

    assert (script_gen.score_hook(better, body=body)["score"]
            > script_gen.score_hook(shipped, body=body)["score"])


def test_a_digit_is_no_longer_the_biggest_term_in_the_scorer():
    """What the reweighting is for.

    A digit appears in 24% of the 55 candidates this pass has generated and in
    42% of the winners - it was the largest discriminating term, and it decided
    hooks on its own. It is now priced with the other virtues, not above them.
    """
    number = script_gen.score_hook(
        "The ocean gains 4 billion tons of salt a year."
    )["components"]["a number"]
    opener = script_gen.score_hook(
        "Nobody has counted the salt in this ocean."
    )["components"]["strong opener"]
    person = script_gen.score_hook(
        "Rain is the reason your ocean tastes of salt."
    )["components"]["second person"]

    assert number <= person
    assert number == pytest.approx(opener, abs=0.01)


def test_being_a_normal_length_is_hygiene_not_a_virtue():
    """80% of generated candidates land in the length band, so paying it like a
    virtue lifted the whole field and left the real terms fighting over what
    was left."""
    length = script_gen.score_hook(
        "Rain is the reason this ocean tastes of salt."
    )["components"]["length"]
    assert length < script_gen.score_hook(
        "The ocean gains 4 billion tons of salt a year."
    )["components"]["a number"]


def test_a_capital_that_only_starts_a_sentence_is_not_a_named_thing():
    scored = script_gen.score_hook(
        "Rain is the reason your ocean tastes of salt. It has for ages."
    )
    assert "names something" not in scored["components"]


def test_a_real_proper_noun_is_still_a_named_thing():
    assert "names something" in script_gen.score_hook(
        "Rain carries salt into the Pacific every single day."
    )["components"]


# --- what the prompt is asked for, against what the scorer pays for --------

def test_the_prompt_asks_for_the_component_the_scorer_pays_most_for():
    """Direct address is the largest single virtue `score_hook` has, and for a
    long time nothing in the prompt asked for it: 10 of the 135 candidates on
    disk put the viewer in the sentence, and 4 of those 10 won their script -
    a 40% win rate against a field of 16%. The scorer cannot choose what it is
    never shown, so the ask and the weight have to move together.

    Since 2026-09-29 the scorer pays for a "you" only when the line points at
    the screen, so that is what the prompt has to ask for: asking for the kind
    it refunds would fill the field with lines built to lose."""
    template = script_gen._HOOK_PROMPT_TEMPLATE
    assert '"you" together with "this"' in template
    assert '"you are looking at"' in template
    assert "something they stand on, breathe, own" not in template

    plain = script_gen.score_hook("Rain carries salt from the rocks to the sea.")
    viewer = script_gen.score_hook("Rain carries salt from the rocks to this sea you swim in.")
    figure = script_gen.score_hook("Rain carries 50 tonnes of salt to the sea.")

    assert viewer["score"] - plain["score"] > figure["score"] - plain["score"], (
        "the prompt asks for a viewer because the scorer pays more for one "
        "than for a number - re-read the ask if that stops being true")


# --- a "you" that points at nothing on screen ------------------------------

@pytest.mark.parametrize("shipped, passed_over", [
    # glacial till, 2026-09-28: 0.22 of the reel watched
    ("Your front yard might be a crime scene from the Ice Age.",
     "This boulder doesn't belong here; it came from a hundred miles away."),
    # jumping spider, 2026-09-22: 0.25
    ("The milk in your fridge has less protein than a spider's.",
     "This spider feeds its babies milk that it lays like an egg."),
    # Crescent City tsunami, 2026-09-21: 0.17
    ("The ground you are standing on is younger than the trees.",
     "Every few decades, the ocean floor washes ashore right here."),
])
def test_a_you_that_points_at_nothing_on_screen_no_longer_wins(shipped, passed_over):
    """Three shipped openers, each against the subject line it beat.

    Every one of the seven "your X" openers posted after the ripples reel had
    a line like the second in its field and won on the "you" alone. None held
    past 0.35 of the reel; the openers that pointed at the screen are the
    account's biggest reels.
    """
    assert (script_gen.score_hook(passed_over)["score"]
            > script_gen.score_hook(shipped)["score"])
    assert script_gen.HOOK_OFF_SCREEN in script_gen.score_hook(shipped)["components"]


@pytest.mark.parametrize("line", [
    "You are looking at a two-hundred-ton rock that floated.",           # Nepal, 1081
    "You can’t run your fingers through this sand.",                     # ripples, 1852
    "This much pressure would shatter your thumbnail; the beetle is fine.",  # beetle, 704
    "You're watching the Rockies get rebuilt in the Gulf of Mexico.",
])
def test_a_you_that_points_at_the_screen_keeps_its_credit(line):
    """The three openers that broke out all say "you" and all point at the
    footage - the change must not touch any of them. The curly apostrophe is
    how the ripples line actually came back from the model."""
    components = script_gen.score_hook(line)["components"]
    assert components["second person"] == 1.0
    assert script_gen.HOOK_OFF_SCREEN not in components


def test_the_charge_gives_back_exactly_what_the_you_collected():
    """A refund, not a penalty: the line ends where it would have with no
    "you" in it. An opener starting "Your" gives back the strong opener too,
    because that 0.8 was paid for the same word."""
    opening = script_gen.score_hook("Your own cat's family tree just grew a new branch.")
    parts = opening["components"]
    assert parts[script_gen.HOOK_OFF_SCREEN] == -(parts["second person"]
                                                  + parts["strong opener"])

    midline = script_gen.score_hook(
        "In old London, the air was a poison you could actually taste.")
    assert midline["components"][script_gen.HOOK_OFF_SCREEN] == -1.0


def test_second_person_still_fires_on_every_you():
    """The performance ledger has recorded `second person` on any "you" since
    the first reel. Withholding it instead of charging for it would have
    changed what that column means halfway down the file."""
    components = script_gen.score_hook(
        "Your body has a biological circuit breaker for absolute terror."
    )["components"]
    assert "second person" in components
    assert "strong opener" in components


@pytest.mark.parametrize("line", [
    "Your kitchen holds a mineral that formed inside a volcano.",
    "Scientists looking at the seafloor found something your eyes cannot.",
])
def test_that_and_a_bare_looking_at_do_not_count_as_pointing(line):
    """"That" is a conjunction far more often than it points, and "looking at"
    without "you are" is somebody else looking."""
    assert script_gen.HOOK_OFF_SCREEN in script_gen.score_hook(line)["components"]
# --- the hook against the body it is joined to -----------------------------

def test_a_hook_the_body_repeats_is_charged_for_it():
    """The shipped deep-ocean opening, restated one sentence later."""
    body = ("It's our deep ocean. Ninety-five percent of it is unexplored. "
            "We have better maps of the moon.")
    hook = "95% of the biggest living space on Earth is unexplored."

    alone = script_gen.score_hook(hook)
    joined = script_gen.score_hook(hook, body=body)

    assert "the body says it again" not in alone["components"]
    assert "the body says it again" in joined["components"]
    assert joined["score"] < alone["score"]


def test_the_restatement_is_caught_across_digits_and_words():
    """"Ninety-five percent" and "95%" are the same claim, and a model
    respelling its own opening is exactly how the fault arrives."""
    assert script_gen._echoes_body(
        "95% of the biggest living space on Earth is unexplored.",
        "Ninety-five percent of it is unexplored.",
    )


def test_a_short_repeated_phrase_is_emphasis_and_is_left_alone():
    """The Inca reel's opening, and the best thing in that script."""
    assert not script_gen._echoes_body(
        "The Inca built 100,000 miles of road without the wheel.",
        "Without the wheel. It sounds impossible.",
    )


def test_a_thin_sentence_cannot_trip_the_restatement_rule():
    """`_overlap` divides by the smaller side, so a two-content-word sentence
    sharing one word scores 0.50 without restating anything."""
    assert not script_gen._echoes_body(
        "Snailfish live 11 kilometres down, where pressure is extreme.",
        "That's the pressure down there.",
    )


def test_a_body_that_moves_on_costs_the_hook_nothing():
    assert not script_gen._echoes_body(
        "We know more about the surface of Mars than we do this place.",
        "Sunlight dies at 650 feet. Below that it is midnight, always.",
    )


def test_the_candidates_are_scored_against_the_body_they_would_open(monkeypatch):
    """End to end: the restating candidate scores higher on its own and still
    loses, which is the whole point of showing the scorer the script."""
    body = ("It's our deep ocean. Ninety-five percent of it is unexplored. "
            "We have better maps of the moon.")
    fake_hooks(monkeypatch,
               "95% of the biggest living space on Earth is unexplored.",
               "We know more about the surface of Mars than we do this place.")
    report = script_gen.choose_hook("the deep ocean", f"A dull opening here. {body}")

    assert report["chosen"] == (
        "We know more about the surface of Mars than we do this place."
    )


# --- number words --------------------------------------------------------

def test_hyphenated_number_words_fold_onto_their_digits():
    assert script_gen._fold_number_word("ninety-five") == "95"
    assert script_gen._fold_number_word("twenty-one") == "21"
    assert script_gen._fold_number_word("eleven") == "11"


def test_a_multiplicative_compound_is_not_added_up():
    """"one-hundred" is 100, not 101, so it is left alone rather than summed."""
    assert script_gen._fold_number_word("one-hundred") == "one-hundred"


def test_an_ordinary_hyphenated_word_is_left_alone():
    assert script_gen._fold_number_word("well-known") == "well-known"


def test_the_word_rate_tracks_the_speed_the_narration_is_read_at():
    """The two are the same quantity. Written down separately, a pipeline that
    speeds the narration up leaves every script the length it was before, and
    every reel lands short of the duration step 0 asked for."""
    assert script_gen.WORDS_PER_MINUTE == round(
        config.SCRIPT_WORDS_PER_MINUTE * config.TTS_SPEED
    )


def test_a_faster_read_asks_for_more_words_not_a_shorter_reel():
    slow = round(config.SCRIPT_WORDS_PER_MINUTE * 1.0)
    fast = round(config.SCRIPT_WORDS_PER_MINUTE * 1.2)
    assert fast > slow
    # both land on the same 40 seconds, which is the point
    assert round(40 * slow / 60 * 60 / slow) == 40
    assert round(40 * fast / 60 * 60 / fast) == 40


def test_hype_words_are_worth_less_than_nothing():
    scored = script_gen.score_hook(
        "An amazing incredible journey into a fascinating world."
    )
    assert scored["components"]["vague words"] < 0
    assert scored["score"] < 0


def test_a_hook_that_runs_on_into_a_second_line_still_scores_worse():
    """The old rule's intent, kept - but paid for in words rather than twice.

    A hook that has eaten the line after it is longer than an opening should
    be, and the length rules already charge for that. What changed is that a
    hook which stays inside an opening's worth of words is no longer charged
    for the full stop in the middle of it.
    """
    one = script_gen.score_hook("Rain is the reason your ocean tastes of salt.")
    two = script_gen.score_hook(
        "Rain is the reason your ocean tastes of salt. It has for ages."
    )
    assert two["score"] < one["score"]
    assert "more than one sentence" not in two["components"]


def test_extra_sentences_are_charged_once_the_hook_is_over_the_ceiling():
    over = " ".join(["word"] * (config.HOOK_MAX_WORDS + 2)) + ". And more."
    assert "more than one sentence" in script_gen.score_hook(over)["components"]


def test_a_third_sentence_is_a_paragraph_even_inside_the_ceiling():
    three = script_gen.score_hook("Salt built up. Rivers carried it. The sea kept it.")
    assert "more than two sentences" in three["components"]


def test_the_setup_then_turn_is_no_longer_buried():
    """The real candidate this rule was changed for.

    It came last of five at -1.8, entirely on the sentence penalty, while a
    flat statement of fact won. Two short sentences are how a reel opens.
    """
    turn = script_gen.score_hook(
        "The deep sea should crush every creature flat. It doesn't. Here's why."
    )
    flat = script_gen.score_hook("The deep sea is crushing.")

    assert turn["score"] > 0
    assert turn["score"] > flat["score"]
    assert "turns on itself" in turn["components"]


def test_a_question_answered_in_the_same_breath_is_not_charged_as_two_sentences():
    # A real candidate from the sleep reel. It used to lose to the sentence
    # penalty, and that penalty no longer touches it. It used to beat "We are
    # getting less sleep than our Stone Age ancestors did." as well, and since
    # 2026-09-29 it does not: "your six hours" points at nothing on screen, so
    # its 1.8 of direct address is given back. See
    # test_a_you_that_points_at_nothing_on_screen_no_longer_wins.
    asked = script_gen.score_hook(
        "Your six hours of sleep? That's barely half what a gorilla gets."
    )
    assert "more than one sentence" not in asked["components"]
    assert "more than two sentences" not in asked["components"]
    assert "a number" in asked["components"]
    assert asked["score"] > 0


def test_a_trailing_clause_is_not_a_turn():
    # Same length and same position as "It doesn't", and it only continues the
    # sentence before it instead of reversing it.
    scored = script_gen.score_hook(
        "Rain is the reason your ocean tastes of salt. It has for ages."
    )
    assert "turns on itself" not in scored["components"]


def test_a_turn_needs_a_setup_longer_than_itself():
    assert "turns on itself" not in script_gen.score_hook("No. Yes.")["components"]


def test_running_past_the_word_ceiling_costs_more_the_further_it_runs():
    limit = config.HOOK_MAX_WORDS
    just_over = " ".join(["word"] * (limit + 1)) + "."
    well_over = " ".join(["word"] * (limit + 6)) + "."

    assert (script_gen.score_hook(well_over)["score"]
            < script_gen.score_hook(just_over)["score"])


def test_an_empty_candidate_scores_below_anything_real():
    assert script_gen.score_hook("")["score"] < -50
    assert script_gen.score_hook("   ")["score"] < -50


def test_the_score_is_the_sum_of_its_reported_parts():
    scored = script_gen.score_hook("Nobody has counted the salt in your ocean.")
    assert scored["score"] == pytest.approx(sum(scored["components"].values()), abs=0.01)


# --- the on-screen short form ----------------------------------------------

def test_a_card_short_enough_already_is_left_alone():
    assert script_gen.hook_card_text("Nobody has seen it") == "Nobody has seen it"


def test_a_card_is_cut_to_the_word_limit():
    card = script_gen.hook_card_text(
        "The ocean gains four billion tons of salt every single year"
    )
    assert len(card.split()) <= config.HOOK_CARD_MAX_WORDS


def test_a_card_stops_at_a_clause_break_when_one_fits():
    card = script_gen.hook_card_text("Salt never leaves, it only arrives", max_words=5)
    assert card == "Salt never leaves"


def test_a_card_never_ends_on_dangling_punctuation():
    for text in ("Nobody has seen it.", "Salt never leaves,", "One thing:"):
        assert not script_gen.hook_card_text(text).endswith((".", ",", ":", ";"))


def test_a_card_never_ends_on_a_word_that_cannot_end_a_thought():
    """The shipped cover of the deep-ocean reel, and the alt text under it.

    Six words of "The quietest place on earth isn't what you think." is the
    line interrupted, not a short version of it.
    """
    assert script_gen.hook_card_text(
        "The quietest place on earth isn't what you think."
    ) == "The quietest place on earth"


def test_the_dangling_trim_only_shortens_and_never_empties():
    """A card trimmed to one word says less than the awkward cut it replaced."""
    assert script_gen.hook_card_text("Nobody is") == "Nobody is"
    assert script_gen.hook_card_text("It was") == "It was"
    for hook in ("The quietest place on earth isn't what you think.",
                 "A spider's eight legs are also its eight ears.",
                 "The silence you're hearing is actually two sounds now."):
        card = script_gen.hook_card_text(hook)
        assert len(card.split()) >= 2
        assert hook.lower().startswith(card.lower())


def test_a_clause_break_still_wins_over_the_dangling_trim():
    """A word before a comma is already somewhere the line breathes."""
    assert script_gen.hook_card_text(
        "It is not what you think, and the reason is stranger", max_words=6
    ) == "It is not what you think"


def test_a_card_never_ends_on_a_bare_figure():
    """The shipped Chernobyl card. A number is the start of a quantity and
    never the end of a thought, so a card ending on one has been guillotined
    mid-measurement - and popping it uncovers the "in" that was holding it up.
    """
    assert script_gen.hook_card_text(
        "They entombed the reactor in 400,000 cubic meters of concrete."
    ) == "They entombed the reactor"


def test_a_figure_with_its_unit_is_left_alone():
    """The rule is about a number cut off from what it counts, not about
    numbers. Both of these are finished thoughts and both must survive."""
    assert script_gen.hook_card_text(
        "The Inca built 100,000 miles of road without the wheel."
    ) == "The Inca built 100,000 miles"
    assert script_gen.hook_card_text(
        "The ocean contains 50 quadrillion tons of salt."
    ) == "The ocean contains 50 quadrillion tons"


def test_a_magnitude_word_is_not_a_bare_figure():
    """"millions" ends on a noun and reads fine where "400,000" cannot.
    Measured over the 29 openings on disk: a magnitude rule fires once, on a
    card that is good."""
    assert script_gen.hook_card_text(
        "This baked sand holds millions of years of rain."
    ) == "This baked sand holds millions"


def test_the_bare_figure_trim_still_respects_the_floor():
    """Only ever shortens, and never below the two-word floor."""
    assert script_gen.hook_card_text("Only 400,000") == "Only 400,000"


def test_a_spelled_out_figure_is_a_bare_figure_too():
    """The shipped case, from the sunflower sea star reel on 2026-09-11.

    Step 2 writes what step 3 has to pronounce, so a narration script states
    its quantities in words and never in digits - which made the digit rule
    unable to fire on the only text it is ever given. The card went onto the
    cover and into the alt text having dropped "arms", the noun the hook turns
    on.
    """
    assert script_gen.hook_card_text(
        "This is a monster with twenty-four arms."
    ) == "This is a monster"
    assert script_gen.hook_card_text(
        "The storm dropped rain for ninety straight hours."
    ) == "The storm dropped rain"


def test_a_spelled_out_figure_with_its_unit_is_left_alone():
    """Same rule as the digit case: a number cut off from what it counts, not
    numbers. Both of these finish their measurement inside the cap."""
    assert script_gen.hook_card_text(
        "This glacier has retreated ninety metres every year."
    ) == "This glacier has retreated ninety metres"


def test_one_is_a_pronoun_before_it_is_a_number():
    """"one" folds to a digit and is far more often a word that ends a thought
    cleanly. Trimming it would have been the fold's worst false positive."""
    assert not script_gen._is_bare_figure("one")
    assert script_gen.hook_card_text(
        "There is no one left to hear it."
    ) == "There is no one left"
    assert script_gen.hook_card_text(
        "This whole forest grew from one seed."
    ) == "This whole forest grew from one"


def test_a_spelled_out_magnitude_word_is_not_a_bare_figure():
    """The fold reaches "million" as readily as "twenty-four", and reinstating
    the magnitude rule by that route is what `_HOOK_CARD_FIGURE_EXCEPT` is
    for. The 2026-09-09 measurement rejected that rule; it stays rejected."""
    for word in ("hundred", "thousand", "million", "billion"):
        assert not script_gen._is_bare_figure(word)
    assert script_gen.hook_card_text(
        "This cave system holds a million bats."
    ) == "This cave system holds a million"


class _Usage:
    """What the SDK hands back on a response."""

    def __init__(self, prompt=0, candidates=0, thoughts=None):
        self.prompt_token_count = prompt
        self.candidates_token_count = candidates
        if thoughts is not None:
            self.thoughts_token_count = thoughts


def test_thinking_tokens_count_as_output():
    """Gemini reports thinking separately and does not fold it into
    candidates_token_count, but bills it at the output rate. Counting only the
    candidates would have under-reported the 2026-09-11 run by ~87%."""
    meter = script_gen._UsageMeter()
    meter.record("gemini-2.5-pro", _Usage(prompt=1500, candidates=400, thoughts=2600))
    assert meter.input_tokens == 1500
    assert meter.output_tokens == 3000       # 400 + 2600
    assert meter.thinking_tokens == 2600
    assert meter.calls == 1


def test_a_response_without_usage_is_not_counted_as_a_call():
    """A call that came back without usage metadata still happened, but
    nothing is known about it - and inventing a zero would read as a free
    call rather than an unmeasured one."""
    meter = script_gen._UsageMeter()
    meter.record("gemini-2.5-pro", None)
    assert meter.calls == 0
    assert meter.as_dict()["input_tokens"] == 0


def test_a_usage_object_missing_fields_does_not_break_a_paid_run():
    """The call is already paid for by the time this runs, so a renamed or
    absent field must never raise."""
    meter = script_gen._UsageMeter()
    meter.record("gemini-2.5-pro", _Usage(prompt=100))   # no thoughts attribute
    assert meter.output_tokens == 0
    assert meter.thinking_tokens == 0
    assert meter.calls == 1


def test_usage_totals_split_by_model_and_reset():
    meter = script_gen._UsageMeter()
    meter.record("gemini-2.5-pro", _Usage(prompt=10, candidates=5))
    meter.record("gemini-2.5-flash", _Usage(prompt=20, candidates=7))
    data = meter.as_dict()
    assert data["calls"] == 2 and data["input_tokens"] == 30
    assert set(data["by_model"]) == {"gemini-2.5-pro", "gemini-2.5-flash"}
    assert data["by_model"]["gemini-2.5-flash"]["input_tokens"] == 20
    meter.reset()
    assert meter.as_dict()["calls"] == 0


def test_an_empty_hook_yields_an_empty_card():
    assert script_gen.hook_card_text("") == ""
    assert script_gen.hook_card_text(None) == ""


# --- splitting and replacing the opening -----------------------------------

def test_the_first_sentence_is_split_off_from_the_rest():
    head, rest = script_gen.split_first_sentence(
        "Imagine total darkness. Sunlight dies at 650 feet. So they glow."
    )
    assert head == "Imagine total darkness."
    assert rest == "Sunlight dies at 650 feet. So they glow."


def test_a_full_stop_in_an_abbreviation_does_not_end_the_sentence():
    head, _ = script_gen.split_first_sentence("Dr. Ono went deeper. Then deeper.")
    assert head == "Dr. Ono went deeper."


def test_a_script_with_no_ending_is_all_opening():
    head, rest = script_gen.split_first_sentence("just one line with no stop")
    assert head == "just one line with no stop"
    assert rest == ""


def test_applying_a_hook_replaces_the_opening_rather_than_prepending_it():
    script = "Imagine total darkness. Sunlight dies at 650 feet."
    result = script_gen.apply_hook(script, "Nobody has seen this place.")

    assert result.startswith("Nobody has seen this place.")
    assert "Imagine total darkness" not in result
    assert "Sunlight dies at 650 feet." in result


def test_applying_a_hook_keeps_the_scripts_paragraph_shape():
    """A script written in paragraphs stays in paragraphs - step 4 reads that
    shape when it cuts the script into beats."""
    paragraphs = "One thing here.\n\nTwo here. Three here."
    assert (script_gen.apply_hook(paragraphs, "New open.")
            == "New open.\n\nTwo here. Three here.")

    flat = "One thing here. Two here."
    assert script_gen.apply_hook(flat, "New open.") == "New open. Two here."


def test_a_bridge_replaces_the_sentence_written_for_the_old_opening():
    """The shipped salt reel's seam: a subordinate clause left attached to a
    sentence that had been deleted out from under it."""
    script = ("The ocean contains 50 quadrillion tons of salt. Especially when "
              "all the water that flows into it is fresh. Rain falls.")
    result = script_gen.apply_hook(
        script,
        "We know more about Mars than this place.",
        "And every river that feeds it is fresh water.",
    )

    assert "Especially when" not in result
    assert result.startswith("We know more about Mars than this place.")
    assert "And every river that feeds it is fresh water." in result
    assert result.endswith("Rain falls."), "the rest of the script survives"


def test_a_bridge_keeps_the_paragraph_shape_too():
    script = ("The ocean holds a lot of salt.\n\nEspecially when the rivers "
              "are fresh. Rain falls here.")
    result = script_gen.apply_hook(script, "A new opening line here.",
                                   "And yet the rivers are fresh.")
    assert result == ("A new opening line here.\n\nAnd yet the rivers are "
                      "fresh. Rain falls here.")


def test_no_bridge_leaves_sentence_two_exactly_as_it_was():
    script = "The ocean holds salt aplenty. Especially when fresh. Rain falls."
    assert "Especially when fresh." in script_gen.apply_hook(
        script, "A new opening line here."
    )


# --- what the bridge is allowed to be --------------------------------------

def test_a_bridge_that_drops_a_number_is_refused():
    """"Keep any fact it carries" is asked for in the prompt, so it is checked
    here - a deleted figure is the one failure invisible in the finished reel."""
    assert script_gen._usable_bridge(
        "Almost none of it has ever been charted.",
        "Some 95 percent of it is unexplored.",
    ) == ""


def test_a_bridge_that_keeps_the_number_is_allowed():
    assert script_gen._usable_bridge(
        "And 95 percent of it is still uncharted.",
        "Some 95 percent of it is unexplored.",
    )


def test_a_bridge_that_runs_away_with_the_script_is_refused():
    assert script_gen._usable_bridge(
        "one two three four five six seven eight nine ten eleven twelve",
        "It is fresh water.",
    ) == ""


def test_an_unchanged_bridge_is_no_bridge_at_all():
    assert script_gen._usable_bridge("It is fresh.", "It is fresh.") == ""
    assert script_gen._usable_bridge("", "It is fresh.") == ""


def test_the_chosen_bridge_reaches_the_script(monkeypatch):
    payload = json.dumps([{
        "hook": "We know more about Mars than this place.",
        "card": "More than Mars",
        "bridge": "And every river that feeds it is fresh.",
    }])
    monkeypatch.setattr(script_gen, "call_gemini", lambda *a, **k: payload)
    report = script_gen.choose_hook(
        "salt",
        "A dull opening sentence here. Especially when the rivers are fresh. "
        "Rain falls on rocks.",
    )

    assert report["replaced"] is True
    assert report["bridge"] == "And every river that feeds it is fresh."
    assert "Especially when" not in script_gen.apply_hook(
        "A dull opening sentence here. Especially when the rivers are fresh. "
        "Rain falls on rocks.",
        report["chosen"], report["bridge"],
    )


def test_an_empty_hook_leaves_the_script_alone():
    script = "Imagine total darkness. Sunlight dies."
    assert script_gen.apply_hook(script, "") == script


# --- choosing between candidates -------------------------------------------

def fake_hooks(monkeypatch, *hooks, card: str = ""):
    """Stand in for the Gemini call with a fixed set of candidates."""
    payload = json.dumps([{"hook": h, "card": card or h} for h in hooks])
    monkeypatch.setattr(script_gen, "call_gemini", lambda *a, **k: payload)


def test_the_best_scoring_candidate_is_the_one_that_gets_used(monkeypatch):
    fake_hooks(monkeypatch,
               "Have you ever wondered about salt?",
               "Rain is the reason your ocean tastes of salt.")
    report = script_gen.choose_hook("salt", "Something dull happens here. Then more.")

    assert report["replaced"] is True
    assert report["chosen"] == "Rain is the reason your ocean tastes of salt."
    assert report["candidates"][0]["text"] == report["chosen"]


def test_a_batch_that_beats_nothing_leaves_the_script_alone(monkeypatch):
    strong = "Rain is the reason your ocean tastes of salt."
    fake_hooks(monkeypatch, "Have you ever wondered?", "An amazing journey awaits.")
    report = script_gen.choose_hook("salt", f"{strong} Then more happens.")

    assert report["replaced"] is False
    assert report["chosen"] is None
    assert any("beat the scriptwriter" in w for w in report["warnings"])


def test_the_incumbent_is_scored_on_the_same_scale_as_the_candidates(monkeypatch):
    opening = "Rain is the reason your ocean tastes of salt."
    fake_hooks(monkeypatch, "Have you ever wondered?")
    report = script_gen.choose_hook("salt", f"{opening} More.")

    assert report["original"] == opening
    assert report["original_score"] == script_gen.score_hook(opening)["score"]


def test_the_hook_pass_can_be_switched_off_without_calling_anything(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("the model must not be called with HOOK_MODE=off")

    monkeypatch.setattr(script_gen, "call_gemini", explode)
    report = script_gen.choose_hook("salt", "An opening. A body.", mode="off")

    assert report["mode"] == "off"
    assert report["replaced"] is False
    assert report["card"], "a card is still derived from the original opening"


def test_a_failed_hook_call_costs_a_warning_and_nothing_else(monkeypatch):
    def fail(*args, **kwargs):
        raise script_gen.ScriptGenerationError("503 unavailable")

    monkeypatch.setattr(script_gen, "call_gemini", fail)
    report = script_gen.choose_hook("salt", "An opening here. A body here.")

    assert report["mode"] == "failed"
    assert report["replaced"] is False
    assert any("503" in warning for warning in report["warnings"])


def test_unparseable_model_output_is_survived(monkeypatch):
    monkeypatch.setattr(script_gen, "call_gemini", lambda *a, **k: "not json at all")
    report = script_gen.choose_hook("salt", "An opening here. A body here.")

    assert report["replaced"] is False
    assert report["warnings"]


def test_the_candidate_list_tolerates_fences_and_bare_strings():
    parsed = script_gen._parse_hook_json(
        '```json\n["A bare string hook.", {"hook": "An object hook.", '
        '"card": "Object card"}]\n```'
    )
    assert [entry["hook"] for entry in parsed] == ["A bare string hook.",
                                                   "An object hook."]
    assert parsed[1]["card"] == "Object card"


def test_a_candidate_card_is_held_to_the_limit_whatever_the_model_returned(monkeypatch):
    long_card = "one two three four five six seven eight nine ten"
    payload = json.dumps([{"hook": "Rain is why your ocean is salty.",
                           "card": long_card}])
    monkeypatch.setattr(script_gen, "call_gemini", lambda *a, **k: payload)
    result = script_gen.generate_hooks("salt", "Dull opening. Body.")

    assert len(result["candidates"][0]["card"].split()) <= config.HOOK_CARD_MAX_WORDS


def test_generated_hooks_go_through_the_same_scrubber_as_the_narration(monkeypatch):
    payload = json.dumps([{"hook": "[excited] **Rain** is why it is salty. ✨"}])
    monkeypatch.setattr(script_gen, "call_gemini", lambda *a, **k: payload)
    result = script_gen.generate_hooks("salt", "Dull opening. Body.")

    text = result["candidates"][0]["text"]
    assert "[" not in text and "*" not in text and "✨" not in text


def test_generate_script_carries_the_hook_and_its_card(monkeypatch):
    monkeypatch.setattr(
        script_gen, "call_gemini",
        lambda prompt, **kwargs: (
            json.dumps([{"hook": "Rain is why your ocean tastes of salt.",
                         "card": "Rain made it salty"}])
            if "replacement opening lines" in prompt
            else "Something dull happens here. Then something else does."
        ),
    )
    result = script_gen.generate_script("why the ocean is salty", 30)

    assert result["script"].startswith("Rain is why your ocean tastes of salt.")
    assert result["hook_card"] == "Rain made it salty"
    assert result["hook"]["replaced"] is True
    assert "Then something else does." in result["script"]


def test_generate_script_can_skip_the_hook_pass(monkeypatch):
    calls: list[str] = []

    def record(prompt, **kwargs):
        calls.append(prompt)
        return "Something dull happens here. Then something else does."

    monkeypatch.setattr(script_gen, "call_gemini", record)
    result = script_gen.generate_script("salt", 30, hooks=False, verify=False)

    assert len(calls) == 1, "only the scriptwriter call"
    assert result["script"].startswith("Something dull happens here.")
    assert result["hook"]["replaced"] is False


# ---------------------------------------------------------------------------
# The structure - scoring the shape of a whole script
# ---------------------------------------------------------------------------

#: Closes its loop, turns in the middle, lands short. Scores the ceiling.
SHAPELY = (
    "Every raindrop carries salt to the sea. "
    "Rivers pick up minerals as they run. "
    "They carry them down to the ocean and leave them there. "
    "But the water leaves again as rain, and the salt cannot follow it. "
    "That is why every raindrop ends up making the sea saltier."
)

#: The shape of a real shipped reel: the last line is the first line again,
#: and sentence four is the script handing itself the question.
RESTATES = (
    "Snailfish live 11 kilometres down. "
    "The pressure there is crushing. "
    "Their cells hold a compound that keeps their proteins working. "
    "So how do they manage it? "
    "Some snailfish live eleven kilometres down and are perfectly comfortable."
)

#: Ends somewhere else entirely - and on a call to action, at that.
DRIFTS = (
    "A crimson mountain lion walked into a Californian backyard. "
    "Its coat had a rare pigment fault. "
    "The zoo took it in and gave it a pen. "
    "Vets checked it over twice a week. "
    "Come and see her."
)


def test_an_ending_that_answers_the_opening_closes_the_loop():
    components = script_gen.score_structure(SHAPELY)["components"]

    assert "closes the loop" in components
    assert "a turn mid-reel" in components
    assert "lands on a short line" in components


def test_an_ending_that_repeats_the_opening_is_punished_harder_than_one_that_drifts():
    restates = script_gen.score_structure(RESTATES)["components"]
    drifts = script_gen.score_structure(DRIFTS)["components"]

    assert restates["the ending restates the opening"] < drifts["the ending drifts"]


def test_the_same_number_spelled_out_still_counts_as_a_restatement():
    """The case the rule exists for: a model rewriting its own opening line
    spells the digits out the second time, and "eleven" is "11"."""
    digits = script_gen._content_words("Snailfish live 11 kilometres down.")
    words = script_gen._content_words("Some snailfish live eleven kilometres down.")

    assert "11" in digits and "11" in words
    assert script_gen._overlap(digits, words) == 1.0


def test_exactly_one_of_the_three_endings_ever_fires():
    endings = {"closes the loop", "the ending restates the opening",
               "the ending drifts"}
    for script in (SHAPELY, RESTATES, DRIFTS):
        fired = endings & set(script_gen.score_structure(script)["components"])
        assert len(fired) == 1, f"{fired} fired on {script[:40]!r}"


def test_a_turn_only_counts_once_the_reel_is_under_way():
    """The same word, moved. At sentence one "but" is part of the opening."""
    early = script_gen.score_structure(
        "But rain is the reason the sea is salty. "
        "Rivers carry minerals down to the sea. "
        "The water leaves again as vapour. "
        "The minerals stay behind for ever."
    )
    middle = script_gen.score_structure(
        "Rain is the reason the sea is salty. "
        "Rivers carry minerals down to the sea. "
        "But the water leaves again as vapour. "
        "The minerals stay behind for ever."
    )

    assert "a turn mid-reel" not in early["components"]
    assert "a turn mid-reel" in middle["components"]


def test_a_last_line_too_long_to_land_scores_nothing_for_it():
    long_close = SHAPELY.replace(
        "That is why every raindrop ends up making the sea saltier.",
        "That is why every single raindrop that has ever fallen anywhere on "
        "this planet ends up making the sea very slightly saltier than it was."
    )

    assert "lands on a short line" in script_gen.score_structure(SHAPELY)["components"]
    assert "lands on a short line" not in script_gen.score_structure(long_close)["components"]


def test_a_question_the_script_asks_itself_costs():
    assert script_gen.score_structure(RESTATES)["components"]["asks itself a question"] == -1.0


def test_a_question_with_no_run_up_is_left_alone():
    """"So how do they manage it?" is the script handing itself the baton.
    "Why does the salt stay behind?" is a question the viewer also has."""
    real = script_gen.score_structure(
        "Rain is the reason the sea is salty. "
        "Rivers carry minerals down to the sea. "
        "Why does the salt stay behind? "
        "The water leaves as vapour and the salt is too heavy to go with it."
    )

    assert "asks itself a question" not in real["components"]


def test_every_filler_question_is_charged_for():
    two = script_gen.score_structure(
        "Rain is the reason the sea is salty. "
        "So how does that work? "
        "Rivers carry minerals down to the sea. "
        "And why does the salt stay? "
        "The water leaves as vapour and the salt cannot follow the rain."
    )

    assert two["components"]["asks itself a question"] == -2.0


# --- a script that argues with itself ---------------------------------------
#
# The case is verbatim from the reel that prompted the check: a script that
# closed its loop, landed on a short line, scored 2.5 and shipped with two
# consecutive sentences saying the same thing in opposite directions.

SEESAW = (
    "We are getting less sleep than our Stone Age ancestors did. "
    "The Hadza hunter-gatherers of Tanzania average six hours a night. "
    "So do the San of Namibia. "
    "But they don't get less sleep than the rest of us. "
    "We get less sleep than them. "
    "Because our ancestors took naps."
)


def test_a_sentence_followed_by_itself_negated_is_charged_for():
    scored = script_gen.score_structure(SEESAW)

    assert scored["components"]["argues with itself"] == -2.0
    assert scored["score"] < config.STRUCTURE_MIN_SCORE   # so it buys a rewrite


def test_the_seesaw_is_found_between_the_two_sentences_that_cause_it():
    sentences = script_gen.split_sentences(SEESAW)
    pairs = script_gen.seesaw_pairs(sentences)

    assert pairs == [(3, 4)]
    assert "don't get less sleep" in sentences[3]
    assert "We get less sleep than them" in sentences[4]


def test_a_flat_claim_knocked_down_in_three_words_is_not_a_seesaw():
    """The good version of the same move must survive.

    "It doesn't." is one of the strongest turns available and carries no
    content words at all, which is exactly what separates it from a
    restatement with the polarity flipped.
    """
    good = (
        "The deep sea should crush every creature flat. "
        "It doesn't. "
        "Snailfish hold their shape eleven kilometres down, "
        "because a molecule called TMAO braces every protein they have. "
        "The pressure never wins."
    )
    assert script_gen.seesaw_pairs(script_gen.split_sentences(good)) == []
    assert "argues with itself" not in script_gen.score_structure(good)["components"]


def test_a_real_contrast_between_two_subjects_is_not_a_seesaw():
    # Two different subjects held against each other share too little to trip
    # the threshold, however emphatic the "not" is.
    contrast = (
        "Most shallow-water fish cannot survive below a kilometre. "
        "Snailfish do not care about the pressure at eleven. "
        "Their cells are built differently."
    )
    assert script_gen.seesaw_pairs(script_gen.split_sentences(contrast)) == []


def test_two_negated_sentences_in_a_row_are_not_a_flip():
    # Both negated is emphasis, not a contradiction - there is no flip.
    doubled = (
        "The pressure does not crush them. "
        "It does not even slow them down. "
        "They simply live there."
    )
    assert script_gen.seesaw_pairs(script_gen.split_sentences(doubled)) == []


def test_the_rewrite_is_told_that_the_script_argues_with_itself():
    faults = script_gen.structure_faults(
        script_gen.score_structure(SEESAW)["components"]
    )
    assert "polarity flipped" in faults


def test_no_shipped_script_trips_the_seesaw_check():
    """Guards the threshold against being tightened into a false positive."""
    for script in (SHAPELY, RESTATES, DRIFTS):
        assert script_gen.seesaw_pairs(script_gen.split_sentences(script)) == []


#: The shape of the canyon reel as it shipped: sentence three says what
#: sentence two just said, in fewer words and with nothing added. No polarity
#: flips, which is exactly why the seesaw check never saw it.
SAYS_IT_TWICE = (
    "This desert was once a seabed. "
    "A canyon is a book that records that deep time. "
    "It's a record of deep time. "
    "Each layer of rock is a page. "
    "But the river did not pile that rock up, it took it away. "
    "You can still read the proof."
)

#: The same script with its repeat taken out - and, crucially, opening on the
#: line it was given. A rewrite that comes back on a different opening is
#: dropped whole, so a fixture that changes one tests the opening guard by
#: accident instead of the rule it was written for.
SAYS_IT_TWICE_REWRITTEN = (
    "This desert was once a seabed. "
    "Layer after layer of shell and mud pressed down into stone. "
    "Each layer holds a different million years. "
    "But the river did not pile that rock up, it took it away. "
    "The seabed is still there, in the wall."
)


def test_a_sentence_reworded_and_said_again_is_caught_without_a_flip():
    """The failure the seesaw check was built too narrowly to see.

    The pair below scores high enough to be charged for, and always did. It
    was never measured, because the seesaw check bailed on any pair whose
    polarity does not turn over before it looked at the overlap at all.
    """
    sentences = script_gen.split_sentences(SAYS_IT_TWICE)
    pairs = script_gen.restatement_pairs(sentences)

    assert pairs == [(1, 2)]
    assert "A canyon is a book" in sentences[1]
    assert "It's a record of deep time" in sentences[2]


def test_the_reworded_repeat_is_charged_hard_enough_to_buy_a_rewrite():
    scored = script_gen.score_structure(SAYS_IT_TWICE)

    assert scored["components"]["says the same thing twice"] == -2.0
    assert scored["score"] < config.STRUCTURE_MIN_SCORE


def test_the_rewrite_is_told_the_second_sentence_is_padding():
    faults = script_gen.structure_faults(
        script_gen.score_structure(SAYS_IT_TWICE)["components"]
    )
    assert "padding" in faults


def test_a_script_that_repeats_itself_is_rewritten_however_well_it_scores(
    monkeypatch,
):
    """The rule had no enforcement: it was in both prompts and nowhere else.

    Dropping the threshold under the script's own score takes away the
    ordinary reason to buy a rewrite, so what is left buying one is the repeat
    itself - rather than the -2.0 charge happening to outweigh whatever else
    fired, which is arithmetic between two constants and not a decision.
    """
    monkeypatch.setattr(config, "STRUCTURE_MIN_SCORE", -10.0)
    fake_rewrite(monkeypatch, SAYS_IT_TWICE_REWRITTEN)
    report = script_gen.choose_structure("canyons", SAYS_IT_TWICE)

    assert report["score"] >= config.STRUCTURE_MIN_SCORE  # nothing is owed
    assert report["revised"] is True                      # and one is bought


def test_a_script_that_says_each_thing_once_still_buys_nothing(monkeypatch):
    # The other half of the test above: it is the repeat that forces the call,
    # not the lowered threshold.
    monkeypatch.setattr(config, "STRUCTURE_MIN_SCORE", -10.0)
    monkeypatch.setattr(script_gen, "call_gemini", _explode)
    report = script_gen.choose_structure("salt", SHAPELY)

    assert report["revised"] is False


def test_a_reworded_repeat_and_a_negated_one_are_not_charged_twice():
    """The two checks split the neighbouring pairs between them on the flip."""
    for script in (SAYS_IT_TWICE, SEESAW):
        sentences = script_gen.split_sentences(script)
        both = set(script_gen.seesaw_pairs(sentences))
        both &= set(script_gen.restatement_pairs(sentences))
        assert both == set()


def test_a_chain_that_carries_a_subject_forward_is_not_a_repeat():
    """Where the threshold has to sit, measured against what has shipped.

    "Then it flows into rivers. The rivers carry the salt into the sea." is
    the closest any legitimate pair in the shipped scripts comes, and it is a
    step in an argument: the second sentence takes the subject the first one
    handed it and does something new with it.
    """
    chain = (
        "Rain picks up carbon dioxide on the way down. "
        "Then it flows into rivers. "
        "The rivers carry the salt into the sea. "
        "Nothing carries it back out."
    )
    assert script_gen.restatement_pairs(script_gen.split_sentences(chain)) == []


def test_a_real_turn_is_still_judged_against_the_higher_threshold():
    """The good version of the flip must not fall through to the lower bar.

    "It doesn't." is one of the strongest turns available. It flips, so it is
    the seesaw check's business and never the restatement check's, and it goes
    on being measured against the threshold that was set for it.
    """
    good = (
        "The deep sea should crush every creature flat. "
        "It doesn't. "
        "Snailfish hold their shape eleven kilometres down. "
        "The pressure never wins."
    )
    sentences = script_gen.split_sentences(good)
    assert script_gen.restatement_pairs(sentences) == []
    assert script_gen.seesaw_pairs(sentences) == []


def test_no_shipped_script_says_anything_twice_by_accident():
    """Guards the lower threshold against being read as a false positive."""
    for script in (SHAPELY, RESTATES, DRIFTS):
        assert script_gen.restatement_pairs(
            script_gen.split_sentences(script)
        ) == []


def test_the_prompt_forbids_saying_a_thing_twice_negated():
    assert "negated" in script_gen.PROMPT_TEMPLATE
    assert "negated" in script_gen._STRUCTURE_PROMPT_TEMPLATE


def test_the_structure_score_is_the_sum_of_its_reported_parts():
    for script in (SHAPELY, RESTATES, DRIFTS):
        scored = script_gen.score_structure(script)
        assert scored["score"] == pytest.approx(sum(scored["components"].values()))


def test_a_one_sentence_script_has_no_shape_to_score():
    scored = script_gen.score_structure("Rain is the reason the sea is salty.")

    assert scored["score"] == 0.0
    assert "too short to have a shape" in scored["components"]


# --- splitting a script into sentences --------------------------------------

def test_an_abbreviation_does_not_start_a_new_sentence():
    assert script_gen.split_sentences(
        "Dr. Ballard found it in 1985. Nobody believed him."
    ) == ["Dr. Ballard found it in 1985.", "Nobody believed him."]


def test_paragraph_breaks_are_flattened_rather_than_counted():
    assert script_gen.split_sentences(
        "Rain carries salt down.\n\nThe sea keeps it for ever."
    ) == ["Rain carries salt down.", "The sea keeps it for ever."]


def test_a_closing_quote_stays_with_the_sentence_it_ends():
    sentences = script_gen.split_sentences(
        'He called it "the salt trap." Nobody uses the name now.'
    )

    assert sentences[0].endswith('"')


def test_the_close_never_swallows_more_than_a_third_of_the_script():
    sentences = [f"This is sentence number {n} of the script." for n in range(4)]

    assert script_gen.closing_sentences(sentences) == sentences[-1:]
    assert script_gen.closing_sentences(sentences * 3) == (sentences * 3)[-3:]


# --- rewriting a script that has no shape -----------------------------------

#: Shares SHAPELY's opening and nothing else: it drifts, and never turns.
SHAPELESS = (
    "Every raindrop carries salt to the sea. "
    "Rivers do their work quietly. "
    "Nobody counts them. "
    "Fish are unbothered by all of it."
)


def fake_rewrite(monkeypatch, script: str):
    """Stand in for the Gemini call with one fixed rewrite."""
    monkeypatch.setattr(script_gen, "call_gemini", lambda *a, **k: script)


def _explode(*args, **kwargs):
    raise AssertionError("the model must not be called here")


def test_a_script_that_already_has_a_shape_buys_no_call(monkeypatch):
    monkeypatch.setattr(script_gen, "call_gemini", _explode)
    report = script_gen.choose_structure("salt", SHAPELY, 20)

    assert report["revised"] is False
    assert report["score"] >= config.STRUCTURE_MIN_SCORE


def test_a_shapeless_script_is_rewritten_when_the_rewrite_scores_better(monkeypatch):
    fake_rewrite(monkeypatch, SHAPELY)
    report = script_gen.choose_structure("salt", SHAPELESS, 20)

    assert report["revised"] is True
    assert report["revised_script"] == SHAPELY
    assert report["revised_score"] > report["score"]


def test_a_rewrite_that_scores_no_better_is_dropped(monkeypatch):
    fake_rewrite(monkeypatch, SHAPELESS)
    report = script_gen.choose_structure("salt", SHAPELESS, 20)

    assert report["revised"] is False
    assert report["revised_script"] == ""
    assert any("kept the original" in w for w in report["warnings"])


def test_a_rewrite_is_scored_even_when_it_is_thrown_away(monkeypatch):
    """The rejected score is on disk beside the kept one, because step 9 can
    only ask whether the pass was worth running if the near misses are there."""
    fake_rewrite(monkeypatch, SHAPELESS)
    report = script_gen.choose_structure("salt", SHAPELESS, 20)

    assert report["revised_score"] is not None
    assert report["revised_components"]


def test_a_rewrite_that_ignores_the_word_budget_is_dropped(monkeypatch):
    fake_rewrite(monkeypatch, SHAPELY)
    report = script_gen.choose_structure("salt", SHAPELESS, 45)

    assert report["revised"] is False
    assert any("word budget" in w for w in report["warnings"])


def test_a_rewrite_that_changed_the_opening_is_dropped(monkeypatch):
    """The opening was handed over as fixed. A model that rewrote it anyway has
    thrown away the line the hook pass scored five candidates to choose - and
    written every sentence after it to follow the one it deleted, which is why
    the whole rewrite goes rather than the opening being pasted back over it.
    The Chernobyl reel shipped on a paste: the body kept the bridge the hook
    pass had written for a different second sentence out of it, and "Not even
    close." went to air answering nothing."""
    fake_rewrite(monkeypatch, SHAPELY.replace(
        "Every raindrop carries salt to the sea.",
        "Something else entirely happens first."))
    report = script_gen.choose_structure("salt", SHAPELESS, 20)

    assert report["revised"] is False
    assert report["revised_script"] == ""
    assert any("changed the opening" in w for w in report["warnings"])


def test_a_script_too_short_to_have_a_middle_is_left_alone(monkeypatch):
    monkeypatch.setattr(script_gen, "call_gemini", _explode)
    report = script_gen.choose_structure(
        "salt", "Rain carries salt down. The sea keeps it. Nothing leaves.", 20)

    assert report["revised"] is False
    assert any("too short" in w for w in report["warnings"])


def test_a_failed_structure_call_costs_a_warning_and_nothing_else(monkeypatch):
    def fail(*args, **kwargs):
        raise script_gen.ScriptGenerationError("503 unavailable")

    monkeypatch.setattr(script_gen, "call_gemini", fail)
    report = script_gen.choose_structure("salt", SHAPELESS, 20)

    assert report["mode"] == "failed"
    assert report["revised"] is False
    assert any("503" in warning for warning in report["warnings"])


def test_an_unusable_rewrite_leaves_the_script_alone(monkeypatch):
    fake_rewrite(monkeypatch, "   ")
    report = script_gen.choose_structure("salt", SHAPELESS, 20)

    assert report["mode"] == "failed"
    assert report["revised"] is False


def test_the_structure_pass_can_be_switched_off_without_calling_anything(monkeypatch):
    monkeypatch.setattr(script_gen, "call_gemini", _explode)
    report = script_gen.choose_structure("salt", SHAPELESS, 20, mode="off")

    assert report["mode"] == "off"
    assert report["revised"] is False
    assert report["components"], "the scoring is free, so it still happens"


def test_the_rewrite_is_told_what_was_wrong_with_the_script(monkeypatch):
    seen = {}

    def capture(prompt, **kwargs):
        seen["prompt"] = prompt
        return SHAPELY

    monkeypatch.setattr(script_gen, "call_gemini", capture)
    script_gen.choose_structure("salt", SHAPELESS, 20)

    assert "the ending has nothing to do with" in seen["prompt"]
    assert "nothing turns in the middle" in seen["prompt"]
    assert "Every raindrop carries salt to the sea." in seen["prompt"]


def test_a_rewrite_goes_through_the_same_scrubber_as_the_narration(monkeypatch):
    fake_rewrite(monkeypatch, "**" + SHAPELY.replace(
        "Rivers pick", "[b-roll: a river] Rivers pick") + "** ✨")
    report = script_gen.choose_structure("salt", SHAPELESS, 20)

    assert "[" not in report["revised_script"]
    assert "*" not in report["revised_script"]
    assert "✨" not in report["revised_script"]


# --- and the whole of step 2 ------------------------------------------------

def _staged_gemini(monkeypatch, calls: list[str]):
    """The four calls step 2 can make, told apart by their prompts.

    Both rewrite passes open on "Rewrite the narration", so they are separated
    on the clause after it. Told apart only by that shared prefix, the
    take-away pass would be handed the structure pass's rewrite, score it the
    same as the incumbent, and reject it - which is a pass doing nothing while
    looking like a pass doing its job.
    """
    # Points at the screen, so the pass still picks it over the incumbent now
    # that a "you" aimed at nothing on screen is refunded.
    hook = "Rain is the reason this ocean you swim in tastes of salt."
    rewritten = (
        f"{hook} "
        "Rivers pick up minerals as they run. "
        "They carry them down and leave them there. "
        "But the water leaves again as vapour, and the salt cannot follow. "
        "That is what every rainfall quietly adds to your ocean."
    )
    handed = (
        f"{hook} "
        "Rivers pick up minerals as they run. "
        "They carry them down and leave them there. "
        "But the water leaves again as vapour, and the salt cannot follow. "
        "Next time you taste sea spray, you are tasting a mountain. "
        "That is what every rainfall quietly adds to your ocean."
    )

    def dispatch(prompt, **kwargs):
        calls.append(prompt)
        if "replacement opening lines" in prompt:
            return json.dumps([{"hook": hook, "card": "Rain made it salty"}])
        if "left holding something" in prompt:
            return handed
        if "Rewrite the narration" in prompt:
            return rewritten
        return ("Salt gets into the sea somehow. "
                "Rivers do their work quietly. "
                "Nobody counts them. "
                "Fish are unbothered by all of it.")

    monkeypatch.setattr(script_gen, "call_gemini", dispatch)
    return hook, rewritten


def test_generate_script_shapes_the_script_around_the_hook_it_chose(monkeypatch):
    """Order matters: the structure pass writes an ending that pays off the
    opening, so it has to see the opening the hook pass settled on."""
    calls: list[str] = []
    hook, rewritten = _staged_gemini(monkeypatch, calls)
    result = script_gen.generate_script("why the ocean is salty", 20,
                                        takeaway=False, verify=False)

    assert len(calls) == 3, "scriptwriter, hook pass, structure pass"
    assert hook in calls[2], "the rewrite was given the chosen hook to keep"
    assert result["script"] == rewritten
    assert result["structure"]["revised"] is True
    assert result["hook"]["replaced"] is True


def test_a_disobedient_rewrite_cannot_cost_the_script_its_bridge(monkeypatch):
    """The Chernobyl failure, end to end.

    The hook pass does not only swap sentence one: it mends sentence two so the
    new opening runs into the body. A rewrite that comes back on its own
    opening has thrown that mend away, and pasting the chosen opening back over
    it cannot put it back - sentence two of the rewrite was written to follow
    the sentence the paste just deleted. So the whole rewrite goes and the
    mended script stands, however well the rewrite scored.

    The shipped reel that proved it opened on a concrete sarcophagus, lost the
    bridge that set up its rebuttal, and said "Not even close." to nothing.
    """
    hook = "Rain is the reason this ocean you swim in tastes of salt."
    bridge = "And that rain is only the start of it."

    def dispatch(prompt, **kwargs):
        if "replacement opening lines" in prompt:
            return json.dumps([{"hook": hook, "card": "Rain made it salty",
                                "bridge": bridge}])
        if "Rewrite the narration" in prompt:
            # Long enough to clear the word band and well enough shaped to
            # beat the incumbent, so the only thing standing between it and
            # the script is the opening it was told to keep.
            return ("Something else entirely happens first. "
                    "Rivers pick up minerals as they run over bare rock and "
                    "old soil. "
                    "They carry that load all the way down and leave it "
                    "standing in the sea. "
                    "But the water leaves again as vapour, and the salt it "
                    "carried cannot follow. "
                    "That is what every rainfall quietly adds to your ocean.")
        return ("Salt gets into the sea somehow. "
                "Rivers do their work quietly. "
                "Nobody counts them. "
                "Fish are unbothered by all of it.")

    monkeypatch.setattr(script_gen, "call_gemini", dispatch)
    result = script_gen.generate_script("why the ocean is salty", 20,
                                        verify=False)

    assert result["hook"]["bridge"] == bridge, "the mend was worth making"
    assert result["structure"]["revised"] is False
    assert result["script"].startswith(hook)
    assert bridge in result["script"], "the mend survived the structure pass"
    assert any("changed the opening" in w
               for w in result["structure"]["warnings"])


def test_generate_script_reports_the_shape_it_left_the_script_in(monkeypatch):
    calls: list[str] = []
    _staged_gemini(monkeypatch, calls)
    report = script_gen.generate_script("salt", 20)["structure"]

    assert report["score"] < config.STRUCTURE_MIN_SCORE
    assert report["revised_score"] > report["score"]
    assert "closes the loop" in report["revised_components"]


def test_generate_script_can_skip_the_structure_pass(monkeypatch):
    calls: list[str] = []
    _staged_gemini(monkeypatch, calls)
    result = script_gen.generate_script("salt", 20, hooks=False,
                                        structure=False, takeaway=False,
                                        verify=False)

    assert len(calls) == 1, "only the scriptwriter call"
    assert result["structure"]["mode"] == "off"
    assert result["structure"]["revised"] is False


def test_a_structure_pass_that_fails_still_yields_a_script(monkeypatch):
    def dispatch(prompt, **kwargs):
        if "Rewrite the narration" in prompt:
            raise script_gen.ScriptGenerationError("503 unavailable")
        return ("Salt gets into the sea somehow. "
                "Rivers do their work quietly. "
                "Nobody counts them. "
                "Fish are unbothered by all of it.")

    monkeypatch.setattr(script_gen, "call_gemini", dispatch)
    result = script_gen.generate_script("salt", 20, hooks=False)

    assert result["script"].startswith("Salt gets into the sea somehow.")
    assert result["structure"]["mode"] == "failed"


# ---------------------------------------------------------------------------
# The take-away
# ---------------------------------------------------------------------------

#: Shaped and paid off, and hands the viewer nothing - which is the whole
#: corpus this pass was built against.
_KEEPS_NOTHING = (
    "The ocean holds more salt than every lake on Earth. "
    "Rivers pick up minerals as they run over bare rock. "
    "They carry them down and leave them in the sea. "
    "But the water leaves again as vapour, and the salt cannot follow. "
    "The ocean has no drain."
)


def test_score_takeaway_pays_for_putting_the_viewer_in_front_of_it():
    script = (_KEEPS_NOTHING
              + " Next time you taste sea spray, you are tasting a mountain.")
    report = script_gen.score_takeaway(script)

    assert "puts the viewer in front of it" in report["components"]
    assert report["score"] >= config.TAKEAWAY_MIN_SCORE


def test_score_takeaway_pays_for_naming_the_thing():
    script = _KEEPS_NOTHING + " The process is called weathering."
    report = script_gen.score_takeaway(script)

    assert "names the thing" in report["components"]


def test_score_takeaway_pays_for_the_viewer_in_the_close():
    script = _KEEPS_NOTHING + " It is the reason your bath water stings."
    report = script_gen.score_takeaway(script)

    assert "the viewer is in the close" in report["components"]


def test_score_takeaway_scores_the_corpus_shape_at_nothing():
    """The measurement the pass exists for, kept as a test.

    A well-shaped script that hands nothing over must score zero. If this
    starts passing on its own, a component has been added that pays for
    something every script already does, and the pass has stopped separating
    anything.
    """
    report = script_gen.score_takeaway(_KEEPS_NOTHING)

    assert report["score"] == 0.0
    assert report["components"] == {}


def test_the_impersonal_you_is_not_a_take_away():
    """"You hear the rumble" is the impersonal "one", not this viewer.

    The deep-ocean reel is the real sentence: "So you don't hear nothing. You
    hear the rumble of a volcano erupting on the other side of the planet."
    Nobody watching is going to hear that volcano. Before the auxiliary was
    required this was the only script in the corpus the rule fired on.
    """
    impersonal = _KEEPS_NOTHING + " You hear the rumble of a volcano."
    personal = _KEEPS_NOTHING + " You'll hear it yourself in any quiet bay."

    assert "puts the viewer in front of it" not in \
        script_gen.score_takeaway(impersonal)["components"]
    assert "puts the viewer in front of it" in \
        script_gen.score_takeaway(personal)["components"]


def test_score_takeaway_survives_an_empty_script():
    report = script_gen.score_takeaway("")

    assert report["score"] == 0.0
    assert "nothing to hand over" in report["components"]


def test_takeaway_faults_name_what_is_missing():
    faults = script_gen.takeaway_faults(
        script_gen.score_takeaway(_KEEPS_NOTHING)["components"])

    assert "never puts the viewer in front of the thing" in faults
    assert "never names what it is describing" in faults


def test_takeaway_faults_say_something_when_everything_fired():
    faults = script_gen.takeaway_faults(
        {"puts the viewer in front of it": 2.0,
         "names the thing": 1.2,
         "the viewer is in the close": 0.8})

    assert faults == "nothing measurable - tighten it"


def test_choose_takeaway_off_never_calls_the_model(monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("the pass is off and called the model anyway")

    monkeypatch.setattr(script_gen, "call_gemini", explode)
    report = script_gen.choose_takeaway("salt", _KEEPS_NOTHING, 20, mode="off")

    assert report["mode"] == "off"
    assert report["revised"] is False
    assert report["score"] == 0.0


def test_choose_takeaway_leaves_a_script_that_already_hands_something_over(
        monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("bought a rewrite for a script that scored")

    monkeypatch.setattr(script_gen, "call_gemini", explode)
    script = (_KEEPS_NOTHING
              + " Next time you taste sea spray, you are tasting a mountain.")
    report = script_gen.choose_takeaway("salt", script, 20, mode="llm")

    assert report["revised"] is False
    assert report["score"] >= config.TAKEAWAY_MIN_SCORE


def test_choose_takeaway_takes_a_rewrite_that_hands_something_over(monkeypatch):
    handed = (
        "The ocean holds more salt than every lake on Earth. "
        "Rivers pick up minerals as they run over bare rock. "
        "They carry them down and leave them in the sea. "
        "But the water leaves again as vapour, and the salt cannot follow. "
        "Next time you taste sea spray, you are tasting a mountain. "
        "The ocean has no drain."
    )
    monkeypatch.setattr(script_gen, "call_gemini",
                        lambda *a, **k: handed)
    report = script_gen.choose_takeaway("salt", _KEEPS_NOTHING, 20, mode="llm")

    assert report["revised"] is True
    assert report["revised_script"] == handed
    assert report["revised_score"] > report["score"]


def test_choose_takeaway_keeps_the_original_when_the_rewrite_gains_nothing(
        monkeypatch):
    monkeypatch.setattr(script_gen, "call_gemini",
                        lambda *a, **k: _KEEPS_NOTHING)
    report = script_gen.choose_takeaway("salt", _KEEPS_NOTHING, 20, mode="llm")

    assert report["revised"] is False
    assert any("kept the original" in w for w in report["warnings"])


def test_a_take_away_may_not_be_bought_by_wrecking_the_ending():
    """The guard the structure pass does not need and this one does.

    Everything this rewrite is asked for lands in the closing sentences, which
    is exactly where `score_structure` measures the payoff. A rewrite that
    hands the viewer something and drops the ending to do it has traded the
    line they would have remembered for the thing they were meant to keep.
    """
    # Hands something over, and ends on a line with nothing to do with the
    # opening - so the take-away score rises and the shape score falls.
    wrecked = (
        "The ocean holds more salt than every lake on Earth. "
        "Rivers pick up minerals as they run over bare rock. "
        "Next time you taste sea spray, you are tasting a mountain. "
        "Anyway, plenty of unrelated weather happens above all of this, and "
        "some of it arrives without any warning whatsoever for anyone. "
        "Clouds do their own thing entirely."
    )
    before = script_gen.score_structure(_KEEPS_NOTHING)["score"]
    after = script_gen.score_structure(wrecked)["score"]
    assert after < before, "the fixture must actually damage the shape"
    assert script_gen.score_takeaway(wrecked)["score"] > \
        script_gen.score_takeaway(_KEEPS_NOTHING)["score"]


def test_choose_takeaway_refuses_a_rewrite_that_damaged_the_shape(monkeypatch):
    wrecked = (
        "The ocean holds more salt than every lake on Earth. "
        "Rivers pick up minerals as they run over bare rock. "
        "Next time you taste sea spray, you are tasting a mountain. "
        "Anyway, plenty of unrelated weather happens above all of this, and "
        "some of it arrives without any warning whatsoever for anyone. "
        "Clouds do their own thing entirely."
    )
    monkeypatch.setattr(script_gen, "call_gemini", lambda *a, **k: wrecked)
    report = script_gen.choose_takeaway("salt", _KEEPS_NOTHING, 20, mode="llm")

    assert report["revised"] is False
    assert any("shape fell" in w for w in report["warnings"])


def test_choose_takeaway_refuses_a_rewrite_off_the_word_budget(monkeypatch):
    monkeypatch.setattr(
        script_gen, "call_gemini",
        lambda *a, **k: ("The ocean holds more salt than every lake on Earth. "
                         "Next time you taste it, you are tasting a mountain."))
    report = script_gen.choose_takeaway("salt", _KEEPS_NOTHING, 20, mode="llm")

    assert report["revised"] is False
    assert any("word budget" in w for w in report["warnings"])


def test_choose_takeaway_leaves_a_script_too_short_to_hand_anything_over(
        monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("rewrote a script with no room to rewrite")

    monkeypatch.setattr(script_gen, "call_gemini", explode)
    report = script_gen.choose_takeaway("salt", "Salt is heavy. It sinks.",
                                        20, mode="llm")

    assert report["revised"] is False
    assert any("too short" in w for w in report["warnings"])


def test_revise_takeaway_drops_a_rewrite_that_changed_the_opening(monkeypatch):
    monkeypatch.setattr(
        script_gen, "call_gemini",
        lambda *a, **k: ("Something else entirely opens this reel now. "
                         "Next time you taste sea spray, you taste a mountain. "
                         "The ocean has no drain."))
    result = script_gen.revise_takeaway("salt", _KEEPS_NOTHING, 20)

    assert result["script"] == ""
    assert any("changed the opening" in w for w in result["warnings"])


def test_a_take_away_pass_that_fails_still_yields_a_script(monkeypatch):
    def dispatch(prompt, **kwargs):
        raise script_gen.ScriptGenerationError("503 unavailable")

    monkeypatch.setattr(script_gen, "call_gemini", dispatch)
    report = script_gen.choose_takeaway("salt", _KEEPS_NOTHING, 20, mode="llm")

    assert report["mode"] == "failed"
    assert report["revised"] is False
    assert any("take-away pass failed" in w for w in report["warnings"])


def test_generate_script_runs_the_take_away_pass_after_the_structure_pass(
        monkeypatch):
    """Order matters the other way round here.

    The take-away pass writes into the ending the structure pass just built,
    so it has to see that ending. Reversed, the structure pass would rewrite
    the hand-over away with nothing watching for it.
    """
    calls: list[str] = []
    _staged_gemini(monkeypatch, calls)
    result = script_gen.generate_script("why the ocean is salty", 20,
                                        verify=False)

    assert len(calls) == 4, "scriptwriter, hook, structure, take-away"
    assert "left holding something" in calls[3]
    assert result["takeaway"]["revised"] is True
    assert "puts the viewer in front of it" in \
        result["takeaway"]["revised_components"]
    assert "Next time you taste sea spray" in result["script"]


def test_generate_script_can_skip_the_take_away_pass(monkeypatch):
    calls: list[str] = []
    _staged_gemini(monkeypatch, calls)
    result = script_gen.generate_script("salt", 20, hooks=False,
                                        structure=False, takeaway=False,
                                        verify=False)

    assert result["takeaway"]["mode"] == "off"
    assert result["takeaway"]["revised"] is False


# ---------------------------------------------------------------------------
# The card and the claim check
# ---------------------------------------------------------------------------

_CARD_BODY = ("Fungal threads link the roots of separate trees. "
              "They trade sugar for minerals along those threads. "
              "But the oldest tree pays in more than sugar. "
              "Next time you walk a wood, look for the stumps still alive. "
              "It feeds its children as it dies.")


def _claim_check_gemini(monkeypatch, *, hook: str, card: str,
                        finding: dict | None, repaired: str | None):
    """Step 2 with a fact-check that rejects the hook and repairs the script.

    The scriptwriter opens on a deliberately flat line so the hook pass
    actually promotes `hook` and carries `card` with it. Offered a candidate
    identical to the one already there, the pass keeps the incumbent - and then
    the card under test is the mechanical cut, not the one the model wrote.
    """
    flat = "Trees are connected underground in some way."
    written = f"{flat} {_CARD_BODY}"
    promoted = f"{hook} {_CARD_BODY}"

    def dispatch(prompt, **kwargs):
        if "replacement opening lines" in prompt:
            return json.dumps([{"hook": hook, "card": card}])
        if "The narration, one sentence per line:" in prompt:
            return json.dumps([finding] if finding else [])
        if "Rewrite the narration" in prompt:      # structure and take-away
            return promoted
        return written

    monkeypatch.setattr(script_gen, "call_gemini", dispatch)
    if repaired is not None:
        monkeypatch.setattr(
            script_gen, "revise_claims",
            lambda script, findings: {"script": repaired, "warnings": []})
    return promoted


def test_a_rejected_claim_does_not_survive_on_the_card(monkeypatch):
    """The forest-roots failure, end to end.

    The card is cut from the opening the hook pass chose, and the claim check
    runs after that - so a claim the checker threw out of the narration was
    still burned over the reel, the cover and the alt text, where it is the
    most-read text in the whole thing.
    """
    hook = "A network smarter than you is under this forest."
    repaired = f"Beneath this forest hides a vast, hidden network. {_CARD_BODY}"
    _claim_check_gemini(
        monkeypatch, hook=hook, card="A network smarter than you",
        finding={"sentence": hook, "kind": "false",
                 "problem": "a mycorrhizal network is not smarter than a human"},
        repaired=repaired)

    result = script_gen.generate_script("how a forest talks through its roots",
                                        40)

    assert result["claims"]["revised"] is True
    assert "smarter than you" not in result["hook_card"]
    assert result["hook_card"] in repaired
    # The hook pass's own report still says what it chose - the card that
    # reaches step 5 is the thing that changed, not the history of it.
    assert result["hook"]["card"] == "A network smarter than you"


def test_the_re_cut_card_is_recorded_as_a_warning(monkeypatch):
    hook = "A network smarter than you is under this forest."
    _claim_check_gemini(
        monkeypatch, hook=hook, card="A network smarter than you",
        finding={"sentence": hook, "kind": "false", "problem": "not true"},
        repaired=f"Beneath this forest hides a hidden network. {_CARD_BODY}")

    result = script_gen.generate_script("forests", 40)

    assert any("card was re-cut" in warning
               for warning in result["claims"]["warnings"])


def test_a_card_is_left_alone_when_the_opening_survived_the_check(monkeypatch):
    """A repair that only touched the body leaves the card exactly as true as
    the sentence it was cut from, and re-cutting would throw away a card the
    model wrote for a mechanical one for nothing."""
    hook = "Under your feet, this forest is talking."
    # The opening is word for word the one the hook pass chose; the repair
    # rewrote the sentence it was told about and nothing else.
    body_fix = (f"{hook} Fungal threads link the roots of separate trees. "
                "Some sugar moves along those threads. "
                "But the oldest tree pays in more than sugar. "
                "Next time you walk a wood, look for the stumps still alive. "
                "It feeds its children as it dies.")
    _claim_check_gemini(
        monkeypatch, hook=hook, card="Forests talk",
        finding={"sentence": "They trade sugar for minerals along those "
                             "threads.",
                 "kind": "false", "problem": "overstated"},
        repaired=body_fix)

    result = script_gen.generate_script("forests", 40)

    assert result["claims"]["revised"] is True
    assert result["hook_card"] == "Forests talk"
    assert not any("card was re-cut"
                   for warning in result["claims"]["warnings"]
                   if "card was re-cut" in warning)


def test_a_card_is_left_alone_when_the_check_found_nothing(monkeypatch):
    hook = "Under your feet, this forest is talking."
    _claim_check_gemini(monkeypatch, hook=hook, card="Forests talk",
                        finding=None, repaired=None)

    result = script_gen.generate_script("forests", 40)

    assert result["claims"]["revised"] is False
    assert result["hook_card"] == "Forests talk"
