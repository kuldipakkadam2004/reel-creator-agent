"""Step 5b - what the model is shown, and what is trusted of what it says back.

The value of this step is entirely in what it refuses to pass on. A verdict
matched to the wrong frame, a problem name nobody else recognises, or a grade
outside the scale would all move a real decision or settle into the ledger as
evidence - so most of what follows is about the model being wrong rather than
about it being right.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

import config  # noqa: E402
from pipeline import look  # noqa: E402


@pytest.fixture(autouse=True)
def a_key_and_a_switch(monkeypatch):
    """Every test here runs with the look on and the key present.

    Both are stubbed rather than read: the key so the suite does not depend on
    whoever's .env it runs beside, the switches so a run with the look turned
    off in config still tests it.
    """
    monkeypatch.setattr(look, "_key_missing", lambda what: [])
    monkeypatch.setattr(config, "LOOK_ENABLED", True)
    monkeypatch.setattr(config, "COVER_LOOK_ENABLED", True)
    monkeypatch.setattr(config, "REEL_LOOK_ENABLED", True)


def answers(payload):
    """Stub the one call, and remember the prompt it was asked with."""
    seen = {}

    def _ask(prompt, *, system_instruction, frames):
        seen["prompt"] = prompt
        seen["system"] = system_instruction
        seen["frames"] = frames
        if isinstance(payload, Exception):
            raise payload
        return payload if isinstance(payload, str) else json.dumps(payload)

    return _ask, seen


def frames(*labels):
    return [(label, b"not really a jpeg") for label in labels]


# ---------------------------------------------------------------------------
# Reading a reply
# ---------------------------------------------------------------------------

def test_a_fenced_reply_is_still_read():
    fenced = "```json\n[{\"frame\": \"shot 0\"}]\n```"
    assert look._parse_json(fenced, "[", "]") == [{"frame": "shot 0"}]


def test_a_reply_with_no_json_in_it_is_refused():
    with pytest.raises(ValueError):
        look._parse_json("I had a look and they all seem fine", "[", "]")


def test_a_grade_is_never_trusted_outside_its_own_scale():
    assert look._clamp_grade(2) == 2
    assert look._clamp_grade("3") == 3
    assert look._clamp_grade(9) == look.GRADE_MAX
    assert look._clamp_grade(-4) == 0
    assert look._clamp_grade(None) == 0
    assert look._clamp_grade("very good") == 0


# ---------------------------------------------------------------------------
# The cover
# ---------------------------------------------------------------------------

def test_each_frame_is_labelled_where_the_model_can_see_it(monkeypatch):
    ask, seen = answers([{"frame": "shot 0", "grade": 2, "ok": True}])
    monkeypatch.setattr(look, "_ask", ask)
    look.judge_covers(frames("shot 0", "shot 4"), topic="deltas")

    # Interleaved and labelled, not a bare list - a verdict read back by
    # position would be a verdict about a different frame.
    assert [label for label, _ in seen["frames"]] == ["shot 0", "shot 4"]
    assert "deltas" in seen["prompt"]


def test_the_verdicts_come_back_under_the_labels_they_went_out_with(monkeypatch):
    ask, _ = answers([
        {"frame": "shot 0", "grade": 3, "ok": True, "why": "a glowing jellyfish"},
        {"frame": "shot 4", "grade": 0, "ok": False, "why": "out of focus"},
    ])
    monkeypatch.setattr(look, "_ask", ask)
    verdicts, notes = look.judge_covers(frames("shot 0", "shot 4"))

    assert notes == []
    assert verdicts["shot 0"] == {"grade": 3, "ok": True,
                                  "why": "a glowing jellyfish"}
    assert verdicts["shot 4"]["ok"] is False


def test_a_verdict_about_a_frame_nobody_sent_is_dropped(monkeypatch):
    ask, _ = answers([
        {"frame": "shot 0", "grade": 2, "ok": True},
        {"frame": "shot 99", "grade": 3, "ok": True},
    ])
    monkeypatch.setattr(look, "_ask", ask)
    verdicts, _ = look.judge_covers(frames("shot 0"))
    assert set(verdicts) == {"shot 0"}


def test_a_frame_the_model_skipped_is_said_out_loud(monkeypatch):
    # It keeps its blind score and races shots that were graded. Unfair, and
    # invisible unless it is written down.
    ask, _ = answers([{"frame": "shot 0", "grade": 2, "ok": True}])
    monkeypatch.setattr(look, "_ask", ask)
    verdicts, notes = look.judge_covers(frames("shot 0", "shot 4"))
    assert set(verdicts) == {"shot 0"}
    assert any("shot 4" in note for note in notes)


def test_a_failed_look_costs_the_verdicts_and_never_the_run(monkeypatch):
    ask, _ = answers(RuntimeError("the model fell over"))
    monkeypatch.setattr(look, "_ask", ask)
    verdicts, notes = look.judge_covers(frames("shot 0"))
    assert verdicts == {}
    assert any("the model fell over" in note for note in notes)


def test_a_reply_of_the_wrong_shape_is_not_a_verdict(monkeypatch):
    ask, _ = answers({"frame": "shot 0", "grade": 3})   # an object, not a list
    monkeypatch.setattr(look, "_ask", ask)
    verdicts, notes = look.judge_covers(frames("shot 0"))
    assert verdicts == {} and notes


def test_nothing_is_asked_when_there_is_nothing_to_look_at(monkeypatch):
    ask, seen = answers([])
    monkeypatch.setattr(look, "_ask", ask)
    assert look.judge_covers([]) == ({}, [])
    assert seen == {}


def test_the_switch_is_honoured_before_the_call_is_made(monkeypatch):
    ask, seen = answers([])
    monkeypatch.setattr(look, "_ask", ask)
    monkeypatch.setattr(config, "COVER_LOOK_ENABLED", False)
    assert look.judge_covers(frames("shot 0")) == ({}, [])
    assert seen == {}


def test_no_key_is_a_skip_with_a_reason_rather_than_a_failure(monkeypatch):
    monkeypatch.setattr(look, "_key_missing",
                        lambda what: [f"{what} was skipped: no key"])
    verdicts, notes = look.judge_covers(frames("shot 0"))
    assert verdicts == {}
    assert notes == ["the cover look was skipped: no key"]


# ---------------------------------------------------------------------------
# The finished reel
# ---------------------------------------------------------------------------

def test_an_ungraded_reel_says_so_rather_than_looking_clean():
    blank = look.blank_grade()
    assert blank["graded"] is False
    assert blank["grade"] is None
    assert blank["problems"] == []


def test_a_graded_reel_carries_its_grade_and_what_it_was_read_from(monkeypatch):
    ask, seen = answers({
        "grade": 2,
        "problems": ["dark"],
        "summary": "solid, a little underexposed at the top",
        "frames": [{"frame": "3.3s", "grade": 2, "shows": "a river mouth"}],
    })
    monkeypatch.setattr(look, "_ask", ask)
    record, notes = look.grade_reel(frames("3.3s", "9.9s"),
                                    topic="deltas", script="A river delta...")

    assert record["graded"] is True
    assert record["grade"] == 2
    assert record["problems"] == ["dark"]
    assert record["frames"] == [{"frame": "3.3s", "grade": 2,
                                 "shows": "a river mouth"}]
    assert notes == []
    assert "A river delta" in seen["prompt"]


def test_a_problem_outside_the_list_is_dropped_and_reported(monkeypatch):
    # Step 9 turns these into facets, and a facet of one reel with a phrasing
    # that never recurs is evidence that can never be read.
    ask, _ = answers({"grade": 2, "problems": ["dark", "slightly washed out"],
                      "summary": "", "frames": []})
    monkeypatch.setattr(look, "_ask", ask)
    record, notes = look.grade_reel(frames("0.5s"))

    assert record["problems"] == ["dark"]
    assert any("slightly washed out" in note for note in notes)


def test_a_problem_named_loosely_is_still_recognised(monkeypatch):
    ask, _ = answers({"grade": 1, "problems": ["Caption Unreadable", "OFF_TOPIC"],
                      "summary": "", "frames": []})
    monkeypatch.setattr(look, "_ask", ask)
    record, _ = look.grade_reel(frames("0.5s"))
    assert record["problems"] == ["caption_unreadable", "off_topic"]


def test_a_grade_worth_a_warning_gets_one(monkeypatch):
    ask, _ = answers({"grade": 1, "problems": ["off_topic"],
                      "summary": "the footage is not about the narration",
                      "frames": []})
    monkeypatch.setattr(look, "_ask", ask)
    record, notes = look.grade_reel(frames("0.5s"))

    assert record["grade"] == 1
    assert any("off_topic" in note and "1/" in note for note in notes)


def test_a_reel_the_model_is_happy_with_is_not_warned_about(monkeypatch):
    ask, _ = answers({"grade": 3, "problems": [], "summary": "clean",
                      "frames": []})
    monkeypatch.setattr(look, "_ask", ask)
    record, notes = look.grade_reel(frames("0.5s"))
    assert record["grade"] == 3 and notes == []


def test_a_failed_grade_ships_the_reel_ungraded(monkeypatch):
    ask, _ = answers(RuntimeError("timed out"))
    monkeypatch.setattr(look, "_ask", ask)
    record, notes = look.grade_reel(frames("0.5s"))

    assert record["graded"] is False
    assert record["problems"] == []
    assert any("ungraded" in note for note in notes)


def test_a_frame_verdict_about_a_timestamp_nobody_sent_is_dropped(monkeypatch):
    ask, _ = answers({"grade": 2, "problems": [], "summary": "",
                      "frames": [{"frame": "99.9s", "grade": 3, "shows": "?"}]})
    monkeypatch.setattr(look, "_ask", ask)
    record, _ = look.grade_reel(frames("0.5s"))
    assert record["frames"] == []


def test_the_problem_list_the_prompt_offers_is_the_one_the_code_accepts(monkeypatch):
    ask, seen = answers({"grade": 3, "problems": [], "summary": "",
                         "frames": []})
    monkeypatch.setattr(look, "_ask", ask)
    look.grade_reel(frames("0.5s"))
    for name in look.PROBLEMS:
        assert name in seen["prompt"]
