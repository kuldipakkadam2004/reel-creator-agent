"""Tests for pipeline/run.py.

Everything here is offline. All six steps are stubbed, so the suite never
scouts a source, calls Gemini, spends an ElevenLabs character or starts an
encode - what is under test is the wiring between them and the decisions step 8
adds: which topic gets made, what happens when a step fails, and what the run
leaves behind for a human who was asleep when it ran.

For a live check run `python pipeline/run.py --dry-run` by hand, which scouts a
real board and chooses from it without making anything. Add --no-phrases and it
costs nothing at all, at the price of a worse choice.

    python -m pytest tests/test_run.py -v
    python tests/test_run.py          # works without pytest installed
"""

from __future__ import annotations

import json
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

import config  # noqa: E402
from pipeline import (assemble, caption, perform, run, script_gen,  # noqa: E402
                      topics, tts, visuals)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def _candidate(topic: str, *, key: str = "", score: float = 10.0,
               duration: int = 40, source: str = "redditrss") -> dict:
    return {"key": key or f"k-{topic}", "topic": topic, "source": source,
            "score": score, "duration": duration}


def _board(*candidates: dict, warnings=None) -> dict:
    board = list(candidates) or [_candidate("why the ocean is salty")]
    return {
        "niche": "ocean",
        "candidates": board,
        "warnings": list(warnings or []),
        "stats": {"shown": len(board)},
    }


class Steps:
    """Stand-ins for the six pipeline steps, with a dial for each failure.

    Records what it was called with, so a test can assert that step 4 was
    handed step 3's dict rather than merely that it ran.
    """

    def __init__(self, tmp_path: Path):
        self.tmp_path = tmp_path
        self.calls: list[str] = []
        self.seen: dict[str, object] = {}
        self.taken: list[tuple[str, str]] = []
        self.boards: list[dict] = []
        self.board = _board()
        # topic -> exception, raised when that topic reaches the step
        self.fail_visuals: dict[str, Exception] = {}
        self.fail: dict[str, Exception] = {}
        #: What step 2's claim check reports it did. "off" is the shape a run
        #: takes when VERIFY_MODE is turned down.
        self.verify_mode = "llm"
        #: What ElevenLabs says is left, or an exception to raise instead.
        #: Plenty by default, so unrelated tests are not measuring the guard.
        self.balance: dict | Exception = {
            "tier": "free", "used": 1000, "limit": 10000, "remaining": 9000,
            "resets": date(2026, 10, 14), "extendable": False}
        self.balance_reads = 0

    # --- the balance guard, before step 0 ------------------------------
    def read_balance(self):
        # Counted apart from `calls`, which is the six steps in order.
        self.balance_reads += 1
        if isinstance(self.balance, Exception):
            raise self.balance
        return dict(self.balance)

    # --- step 0 --------------------------------------------------------
    def find_topics(self, **kwargs):
        self.calls.append("topics")
        self.seen["find_topics_kwargs"] = kwargs
        if isinstance(self.fail.get("topics"), Exception):
            raise self.fail["topics"]
        return self.board

    def save_board(self, board, path=None):
        self.boards.append(board)
        return []

    def take_topic(self, key, topic="", **kwargs):
        self.taken.append((key, topic))
        return []

    # --- step 2 --------------------------------------------------------
    def generate_script(self, topic, duration=None, **kwargs):
        self.calls.append("script")
        self.seen["script_topic"] = topic
        self.seen["script_duration"] = duration
        if isinstance(self.fail.get("script"), Exception):
            raise self.fail["script"]
        return {"topic": topic, "script": f"words about {topic}.",
                "estimated_duration": duration or 40, "hook_card": "HOOK",
                # What step 2 hands back when the fact-check ran and the
                # script came through it clean - the normal case, so unrelated
                # tests are not measuring a run that skipped it.
                "claims": {"mode": self.verify_mode, "findings": [],
                           "revised": False, "blocked": False, "warnings": []}}

    def save_script(self, result, output_path=None):
        stem = "-".join(result["topic"].split()[:3])
        path = self.tmp_path / "scripts" / f"{stem}.txt"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(result["script"], encoding="utf-8")
        return {**result, "script_path": str(path),
                "metadata_path": str(path.with_suffix(".json"))}

    # --- step 3 --------------------------------------------------------
    def generate_audio(self, script, output_path, voice_id=None):
        self.calls.append("tts")
        self.seen["tts_script"] = script
        self.seen["tts_path"] = output_path
        if isinstance(self.fail.get("tts"), Exception):
            raise self.fail["tts"]
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        Path(output_path).write_bytes(b"\xff\xfb\x90\x00")
        return {"audio_path": str(output_path), "duration_seconds": 41.5,
                "character_count": len(script)}

    # --- step 4 --------------------------------------------------------
    def source_visuals(self, script, audio=None, **kwargs):
        self.calls.append("visuals")
        self.seen["visuals_script"] = script
        self.seen["visuals_audio"] = audio
        topic = script["topic"] if isinstance(script, dict) else ""
        if topic in self.fail_visuals:
            raise self.fail_visuals[topic]
        if isinstance(self.fail.get("visuals"), Exception):
            raise self.fail["visuals"]
        return {"run_id": "ocean-1a2b3c", "warnings": ["one clip was small"],
                "assets": [], "beats": [],
                "stats": {"assets": 9, "beats": 5, "from_cache": 2,
                          "downloaded_bytes": 5 * 1024 * 1024}}

    # --- step 5 --------------------------------------------------------
    def assemble_video(self, audio, visuals_manifest, **kwargs):
        self.calls.append("assemble")
        self.seen["assemble_audio"] = audio
        self.seen["assemble_visuals"] = visuals_manifest
        if isinstance(self.fail.get("assemble"), Exception):
            raise self.fail["assemble"]
        run_dir = self.tmp_path / "out" / "2026-08-24" / "ocean-1a2b3c"
        run_dir.mkdir(parents=True, exist_ok=True)
        video = run_dir / "reel.mp4"
        video.write_bytes(b"\x00")
        return {"run_id": "ocean-1a2b3c", "video_path": str(video),
                "duration_seconds": 43.1, "warnings": [],
                "stats": {"shots": 12}}

    # --- step 6 --------------------------------------------------------
    def generate_caption(self, source, **kwargs):
        self.calls.append("caption")
        self.seen["caption_source"] = source
        if isinstance(self.fail.get("caption"), Exception):
            raise self.fail["caption"]
        run_dir = Path(source["video_path"]).parent
        post = run_dir / "post.txt"
        post.write_text("caption", encoding="utf-8")
        return {"post_path": str(post), "warnings": [],
                "hashtags": ["#ocean", "#science"],
                "chars": {"caption": 240}}


@pytest.fixture
def steps(tmp_path, monkeypatch):
    """Stub all six steps and point every path at tmp_path."""
    stub = Steps(tmp_path)

    monkeypatch.setattr(topics, "find_topics", stub.find_topics)
    monkeypatch.setattr(topics, "save_board", stub.save_board)
    monkeypatch.setattr(topics, "take_topic", stub.take_topic)
    monkeypatch.setattr(script_gen, "generate_script", stub.generate_script)
    monkeypatch.setattr(script_gen, "save_script", stub.save_script)
    monkeypatch.setattr(tts, "generate_audio", stub.generate_audio)
    monkeypatch.setattr(tts, "read_balance", stub.read_balance)
    monkeypatch.setattr(visuals, "source_visuals", stub.source_visuals)
    monkeypatch.setattr(assemble, "assemble_video", stub.assemble_video)
    monkeypatch.setattr(caption, "generate_caption", stub.generate_caption)

    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "logs")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(config, "AUDIO_DIR", tmp_path / "audio")
    monkeypatch.setattr(config, "RUN_LEDGER_FILE", tmp_path / "runs.json")
    # Step 9's ledger, for the same reason as every other path here: a run made
    # by the suite must not land in the ledger a real account is measured by.
    monkeypatch.setattr(config, "PERFORMANCE_FILE", tmp_path / "performance.json")
    # ...and the report the read writes from it. These are derived from the
    # real OUTPUT_DIR when config is imported, so moving OUTPUT_DIR above does
    # not move them. Until 2026-10-01 every run of the suite replaced the real
    # output/report.txt, report.json and reports/<today>.txt with a report of
    # this empty ledger: "REELS : 0 known". The reports dated 09-29 and 09-30
    # were both the suite's, and nothing said so.
    monkeypatch.setattr(config, "PERFORMANCE_REPORT_FILE", tmp_path / "report.txt")
    monkeypatch.setattr(config, "PERFORMANCE_REPORT_JSON", tmp_path / "report.json")
    monkeypatch.setattr(config, "PERFORMANCE_REPORT_DIR", tmp_path / "reports")
    monkeypatch.setattr(config, "PERFORMANCE_EXPLORE", "off")
    # Off by default here for the same reason: with a PERFORMANCE_PROVIDER set
    # in a developer's own .env this would put the suite on the network and
    # ask Instagram about a real account once per test.
    monkeypatch.setattr(config, "RUN_FETCH_INSIGHTS", False)
    monkeypatch.setattr(config, "RUN_ONCE_PER_DAY", True)
    monkeypatch.setattr(config, "RUN_TOPIC_ATTEMPTS", 2)
    monkeypatch.setattr(config, "RUN_TOPIC_MIN_SCORE", 0.0)
    monkeypatch.setattr(config, "RUN_MIN_TTS_CHARACTERS", 900)
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("ELEVENLABS_API_KEY", "test-key")
    return stub


# ---------------------------------------------------------------------------
# The happy path
# ---------------------------------------------------------------------------

def test_run_chains_all_six_steps_in_order(steps):
    record = run.run_pipeline()

    assert record["status"] == "ok"
    assert steps.calls == ["topics", "script", "tts", "visuals",
                           "assemble", "caption"]


def test_each_step_is_handed_the_previous_step_s_result(steps):
    run.run_pipeline()

    # step 4 gets step 2's dict and step 3's dict, not paths it re-derived
    assert steps.seen["visuals_script"]["topic"] == "why the ocean is salty"
    assert steps.seen["visuals_audio"]["duration_seconds"] == 41.5
    # step 5 gets step 3's audio and step 4's manifest
    assert steps.seen["assemble_visuals"]["run_id"] == "ocean-1a2b3c"
    # step 6 reads step 5's assembly
    assert steps.seen["caption_source"]["video_path"].endswith("reel.mp4")


def test_record_carries_every_path_the_run_produced(steps):
    record = run.run_pipeline()

    assert record["topic"] == "why the ocean is salty"
    assert record["run_id"] == "ocean-1a2b3c"
    assert record["video_path"].endswith("reel.mp4")
    assert record["post_path"].endswith("post.txt")
    assert record["script_path"].endswith(".txt")
    assert record["audio_path"].endswith(".mp3")
    assert record["duration_seconds"] == 43.1
    assert set(record["steps"]) == set(run.STEPS)


def test_the_length_step_0_chose_reaches_step_2(steps):
    steps.board = _board(_candidate("deep sea vents", duration=55))
    run.run_pipeline()
    assert steps.seen["script_duration"] == 55


def test_an_explicit_duration_beats_the_board_s(steps):
    steps.board = _board(_candidate("deep sea vents", duration=55))
    run.run_pipeline(duration=25)
    assert steps.seen["script_duration"] == 25


def test_narration_is_named_after_the_script(steps):
    run.run_pipeline()
    assert Path(steps.seen["tts_path"]).stem == "why-the-ocean"


def test_warnings_are_labelled_with_the_step_that_raised_them(steps):
    steps.board = _board(warnings=["youtube: no key - skipped"])
    record = run.run_pipeline()

    assert "topics: youtube: no key - skipped" in record["warnings"]
    assert "visuals: one clip was small" in record["warnings"]


def test_a_reel_that_was_never_fact_checked_says_so(steps):
    """"No findings" and "nobody looked" read the same from a log.

    The deep-ocean reel shipped with the claim check off and the log's
    warnings block said nothing at all about it.
    """
    steps.verify_mode = "off"
    record = run.run_pipeline()

    assert any("the fact-check did not run" in w for w in record["warnings"])
    assert any("VERIFY_MODE=off" in w for w in record["warnings"])


def test_a_fact_checked_reel_does_not_carry_the_warning(steps):
    record = run.run_pipeline()
    assert not any("fact-check did not run" in w for w in record["warnings"])


def test_spend_is_recorded(steps):
    record = run.run_pipeline()
    assert record["spend"]["tts_characters"] > 0
    assert record["spend"]["downloaded_bytes"] == 5 * 1024 * 1024


# ---------------------------------------------------------------------------
# Choosing the topic
# ---------------------------------------------------------------------------

def test_the_top_of_the_board_is_what_gets_made(steps):
    steps.board = _board(
        _candidate("first", score=30.0),
        _candidate("second", score=20.0),
    )
    record = run.run_pipeline()
    assert record["topic"] == "first"


def test_the_chosen_topic_is_marked_made_so_it_never_comes_back(steps):
    steps.board = _board(_candidate("first", key="abc"))
    run.run_pipeline()
    assert steps.taken == [("abc", "first")]


def test_a_topic_is_marked_made_before_the_narration_is_paid_for(steps):
    # Step 4 finding no footage must not leave the topic at the top of
    # tomorrow's board, or a daily run buys the same script every evening.
    steps.board = _board(_candidate("unshootable", key="abc"))
    steps.fail_visuals["unshootable"] = visuals.NoAssetsError("no footage")

    record = run.run_pipeline(attempts=1)

    assert record["status"] == "failed"
    assert steps.taken == [("abc", "unshootable")]


def test_an_explicit_topic_does_not_scout_a_board(steps):
    record = run.run_pipeline("hand picked topic")

    assert "topics" not in steps.calls
    assert record["topic"] == "hand picked topic"
    assert record["topic_source"] == "given"
    assert steps.taken == []          # nothing off a board, nothing to mark


def test_min_score_can_refuse_a_weak_board(steps):
    steps.board = _board(_candidate("thin", score=3.0))
    record = run.run_pipeline(min_score=10.0)

    assert record["status"] == "skipped"
    assert "RUN_TOPIC_MIN_SCORE" in record["reason"]
    assert "script" not in steps.calls


def test_choose_candidates_caps_at_the_attempt_limit():
    board = _board(*[_candidate(f"t{i}") for i in range(6)])
    queue, notes = run.choose_candidates(board, attempts=2)
    assert [c["topic"] for c in queue] == ["t0", "t1"]
    assert notes == []


def test_choose_candidates_reports_an_empty_board():
    queue, notes = run.choose_candidates({"candidates": []})
    assert queue == []
    assert "empty" in notes[0]


def test_choose_candidates_notes_what_the_floor_excluded():
    board = _board(_candidate("strong", score=30.0),
                   _candidate("weak", score=1.0))
    queue, notes = run.choose_candidates(board, attempts=5, min_score=10.0)
    assert [c["topic"] for c in queue] == ["strong"]
    assert "under the 10 score floor" in notes[0]


# ---------------------------------------------------------------------------
# Failure policy
# ---------------------------------------------------------------------------

def test_no_footage_falls_through_to_the_next_topic(steps):
    steps.board = _board(_candidate("unshootable"), _candidate("shootable"))
    steps.fail_visuals["unshootable"] = visuals.NoAssetsError("no footage")

    record = run.run_pipeline()

    assert record["status"] == "ok"
    assert record["topic"] == "shootable"
    assert [a["status"] for a in record["attempts"]] == ["failed", "ok"]
    assert record["attempts"][0]["failed_at"] == "visuals"


def test_the_fall_through_stops_at_the_attempt_limit(steps):
    steps.board = _board(*[_candidate(f"t{i}") for i in range(5)])
    steps.fail["visuals"] = visuals.NoAssetsError("no footage anywhere")

    record = run.run_pipeline(attempts=3)

    assert record["status"] == "failed"
    assert len(record["attempts"]) == 3


def test_a_systemic_failure_does_not_burn_a_second_topic(steps):
    # A rejected key fails identically on every candidate, and each retry pays
    # for a script to find that out.
    steps.board = _board(_candidate("first"), _candidate("second"))
    steps.fail["tts"] = tts.TTSAuthError("the API key was rejected")

    record = run.run_pipeline()

    assert record["status"] == "failed"
    assert len(record["attempts"]) == 1
    assert record["reason"].startswith("tts:")


def test_a_dead_ffmpeg_stops_the_run(steps):
    steps.board = _board(_candidate("first"), _candidate("second"))
    steps.fail["assemble"] = assemble.FFmpegNotFoundError("no ffmpeg")

    record = run.run_pipeline()

    assert record["status"] == "failed"
    assert len(record["attempts"]) == 1


def test_a_failed_run_still_reports_what_it_paid_for(steps):
    steps.fail["assemble"] = assemble.FFmpegError("encode died")
    record = run.run_pipeline(attempts=1)

    assert record["status"] == "failed"
    assert record["script_path"]                 # step 2 was billed
    assert record["audio_path"]                  # step 3 was billed
    assert record["spend"]["tts_characters"] > 0


def test_a_caption_failure_leaves_the_reel_standing(steps):
    steps.fail["caption"] = caption.CaptionError("gemini said no")

    record = run.run_pipeline()

    assert record["status"] == "partial"
    assert record["video_path"].endswith("reel.mp4")
    assert record["post_path"] == ""
    assert any("gemini said no" in w for w in record["warnings"])


def test_a_dead_board_fails_the_run_without_spending(steps):
    steps.fail["topics"] = topics.TopicsError("every source failed")

    record = run.run_pipeline()

    assert record["status"] == "failed"
    assert record["reason"].startswith("topics:")
    assert "script" not in steps.calls


def test_a_missing_key_is_caught_before_the_board_is_scouted(steps, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)

    record = run.run_pipeline()

    assert record["status"] == "failed"
    assert "ELEVENLABS_API_KEY" in record["reason"]
    assert steps.calls == []


# ---------------------------------------------------------------------------
# The balance guard
# ---------------------------------------------------------------------------

def test_a_short_balance_stops_the_run_before_the_board_is_scouted(steps):
    steps.balance["remaining"] = 340

    record = run.run_pipeline()

    assert record["status"] == "failed"
    assert record["reason"].startswith("tts:")
    assert "340" in record["reason"] and "RUN_MIN_TTS_CHARACTERS" in record["reason"]
    assert steps.calls == []


def test_a_balance_at_the_floor_is_enough(steps):
    steps.balance["remaining"] = 900

    assert run.run_pipeline()["status"] == "ok"


def test_a_plan_that_bills_past_its_limit_is_never_stopped(steps):
    steps.balance.update(remaining=0, extendable=True)

    assert run.run_pipeline()["status"] == "ok"


def test_an_unreadable_balance_warns_and_carries_on(steps):
    """Step 3 still refuses a real shortfall; a failed read learned nothing."""
    steps.balance = tts.TTSError("ElevenLabs would not report the balance [401]")

    record = run.run_pipeline()

    assert record["status"] == "ok"
    assert record["spend"]["tts_balance"] is None
    assert any(w.startswith("tts: the balance was not checked")
               for w in record["warnings"])


def test_a_floor_of_zero_never_asks(steps, monkeypatch):
    monkeypatch.setattr(config, "RUN_MIN_TTS_CHARACTERS", 0)
    steps.balance["remaining"] = 0

    assert run.run_pipeline()["status"] == "ok"
    assert steps.balance_reads == 0


def test_the_balance_is_asked_after_the_day_guard(steps):
    """A day that already has a reel is skipped without a network call."""
    run.run_pipeline()
    reads = steps.balance_reads

    assert run.run_pipeline()["status"] == "skipped"
    assert steps.balance_reads == reads


def test_a_dry_run_does_not_ask_for_the_balance(steps):
    run.run_pipeline(dry_run=True)

    assert steps.balance_reads == 0


def test_the_summary_says_what_is_left_after_the_run(steps, capsys):
    steps.balance["remaining"] = 2215

    record = run.run_pipeline()
    run._print_summary(record)

    assert record["spend"]["tts_balance"] == 2215
    left = 2215 - record["spend"]["tts_characters"]
    assert f"({left:,} left)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# The once-per-day guard
# ---------------------------------------------------------------------------

def test_a_second_run_on_the_same_day_is_refused(steps):
    first = run.run_pipeline()
    assert first["status"] == "ok"

    second = run.run_pipeline()

    assert second["status"] == "skipped"
    assert "already" in second["reason"]
    assert steps.calls.count("tts") == 1


def test_force_overrides_the_guard(steps):
    run.run_pipeline()
    second = run.run_pipeline(force=True)

    assert second["status"] == "ok"
    assert steps.calls.count("tts") == 2


def test_the_guard_can_be_turned_off(steps):
    run.run_pipeline()
    assert run.run_pipeline(once_per_day=False)["status"] == "ok"


def test_a_failed_run_does_not_hold_the_day(steps):
    steps.fail["assemble"] = assemble.FFmpegError("encode died")
    assert run.run_pipeline(attempts=1)["status"] == "failed"

    steps.fail.clear()
    assert run.run_pipeline()["status"] == "ok"


def test_a_reel_on_disk_counts_even_with_no_ledger(steps, tmp_path):
    day = date.today().strftime(config.VIDEO_RUN_DATE_FORMAT)
    made = tmp_path / "out" / day / "ocean-1a2b3c" / config.VIDEO_FILENAME
    made.parent.mkdir(parents=True, exist_ok=True)
    made.write_bytes(b"\x00")

    found = run.finished_today(ledger={"runs": []})

    assert found is not None
    assert found["run_id"] == "ocean-1a2b3c"


def test_yesterday_s_reel_does_not_block_today(steps):
    yesterday = (date.today() - timedelta(days=1)).strftime(
        config.VIDEO_RUN_DATE_FORMAT)
    run.append_run({"day": yesterday, "status": "ok", "topic": "old news",
                    "spend": {}}, config.RUN_LEDGER_FILE)

    assert run.run_pipeline()["status"] == "ok"


# ---------------------------------------------------------------------------
# The dry run
# ---------------------------------------------------------------------------

def test_a_dry_run_stops_before_anything_is_made(steps, capsys):
    record = run.run_pipeline(dry_run=True)

    assert record["status"] == "skipped"
    assert steps.calls == ["topics"]
    assert record["topic"] == "why the ocean is salty"
    assert "DRY RUN" in capsys.readouterr().out


def test_a_dry_run_previews_the_choice_a_real_run_would_make(steps):
    # Scouting on a cheaper rule would preview a topic the real run is not
    # going to pick, which is worse than not previewing at all.
    run.run_pipeline(dry_run=True)
    kwargs = steps.seen["find_topics_kwargs"]
    assert kwargs["phrases"] is None
    assert kwargs["niche_match"] is None


def test_a_real_run_leaves_step_0_on_its_configured_mode(steps):
    run.run_pipeline()
    kwargs = steps.seen["find_topics_kwargs"]
    assert kwargs["phrases"] is None
    assert kwargs["niche_match"] is None
    assert kwargs["repeats"] is None


def test_no_phrases_asks_step_0_for_its_free_mode(steps):
    run.run_pipeline(dry_run=True, no_phrases=True)
    kwargs = steps.seen["find_topics_kwargs"]
    assert kwargs["phrases"] == "heuristic"
    assert kwargs["niche_match"] == "terms"
    assert kwargs["repeats"] == "terms", "the repeat check is a call too"


def test_no_phrases_says_the_choice_is_not_the_real_one(steps, capsys):
    run.run_pipeline(dry_run=True, no_phrases=True)
    assert "will very likely pick something else" in capsys.readouterr().out


def test_no_phrases_also_applies_to_a_real_run(steps):
    run.run_pipeline(no_phrases=True)
    assert steps.seen["find_topics_kwargs"]["phrases"] == "heuristic"


def test_a_board_trial_arms_step_0_before_the_board_is_built(steps, monkeypatch):
    """The curiosity arm has to be decided before `find_topics`, not after.

    Every other arm takes the topic as given; this one changes which topic
    wins, so a run that chose it where the others are chosen - inside
    `_make_reel`, after the board had already been scored - would record an
    arm it never actually took.
    """
    monkeypatch.setattr(config, "PERFORMANCE_EXPLORE", "curiosity")
    monkeypatch.setattr(perform, "choose_arm",
                        lambda *a, **k: {"name": "curiosity", "arm": "off",
                                         "index": 1})

    record = run.run_pipeline()

    assert steps.seen["find_topics_kwargs"]["curiosity"] is False
    assert record["trial"] == {"name": "curiosity", "arm": "off", "index": 1}


def test_the_control_arm_of_a_board_trial_leaves_step_0_alone(steps, monkeypatch):
    monkeypatch.setattr(config, "PERFORMANCE_EXPLORE", "curiosity")
    monkeypatch.setattr(perform, "choose_arm",
                        lambda *a, **k: {"name": "curiosity", "arm": "on",
                                         "index": 0})

    run.run_pipeline()

    assert "curiosity" not in steps.seen["find_topics_kwargs"],         "the on arm is what every other caller gets, not an explicit True"


def test_a_trial_on_something_else_never_reaches_the_board(steps, monkeypatch):
    monkeypatch.setattr(config, "PERFORMANCE_EXPLORE", "hook")
    monkeypatch.setattr(perform, "choose_arm",
                        lambda *a, **k: {"name": "hook", "arm": "off",
                                         "index": 1})

    record = run.run_pipeline()

    assert "curiosity" not in steps.seen["find_topics_kwargs"]
    assert record["trial"]["name"] == "hook", "and it is still recorded"


def test_a_board_trial_is_announced_where_it_is_spent(steps, monkeypatch, capsys):
    """At step 1, and only there. Announcing it again at step 2 would claim
    the script pass did something differently, and it did not."""
    monkeypatch.setattr(config, "PERFORMANCE_EXPLORE", "curiosity")
    monkeypatch.setattr(perform, "choose_arm",
                        lambda *a, **k: {"name": "curiosity", "arm": "off",
                                         "index": 1})

    run.run_pipeline()
    out = capsys.readouterr().out

    assert out.count('trial "curiosity"') == 1
    assert 'this board takes the "off" arm' in out


def test_a_dry_run_previews_the_arm_it_would_have_taken(steps, monkeypatch):
    """A preview scored on a different board is a preview of nothing."""
    monkeypatch.setattr(config, "PERFORMANCE_EXPLORE", "curiosity")
    monkeypatch.setattr(perform, "choose_arm",
                        lambda *a, **k: {"name": "curiosity", "arm": "off",
                                         "index": 1})

    run.run_pipeline(dry_run=True)

    assert steps.seen["find_topics_kwargs"]["curiosity"] is False


def test_a_dry_run_ignores_the_once_per_day_guard(steps):
    run.run_pipeline()
    assert run.run_pipeline(dry_run=True)["status"] == "skipped"
    assert steps.calls.count("topics") == 2


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

def test_every_run_lands_in_the_ledger(steps):
    run.run_pipeline()
    steps.fail["assemble"] = assemble.FFmpegError("encode died")
    run.run_pipeline(force=True, attempts=1)

    runs = run.load_ledger(config.RUN_LEDGER_FILE)["runs"]

    assert [entry["status"] for entry in runs] == ["ok", "failed"]
    assert runs[0]["topic"] == "why the ocean is salty"
    assert runs[1]["reason"].startswith("assemble:")


def test_a_reason_is_one_line_however_the_error_arrived(steps):
    # ffmpeg attaches several hundred lines of encoder statistics to its
    # exception. The log keeps all of it; the ledger column cannot.
    noise = "\n".join(
        ["ffmpeg exited 3221225786:"]
        + [f"[libx264] statistic line {n}" for n in range(40)]
    )
    steps.fail["assemble"] = assemble.FFmpegError(noise)

    record = run.run_pipeline(attempts=1)
    reason = record["reason"]

    assert "\n" not in reason
    assert len(reason) < 260
    assert "ffmpeg exited 3221225786" in reason
    assert "40 more line(s) in the log" in reason


def test_a_reason_that_is_one_long_line_is_cut(steps):
    steps.fail["assemble"] = assemble.FFmpegError("x" * 400)
    reason = run.run_pipeline(attempts=1)["reason"]

    assert reason.endswith("...")
    assert len(reason) < 200


def test_the_full_error_is_still_kept_on_the_attempt(steps):
    steps.fail["assemble"] = assemble.FFmpegError("line one\nline two")
    record = run.run_pipeline(attempts=1)
    assert record["attempts"][0]["error"] == "line one\nline two"


def test_a_short_error_is_left_alone():
    assert run._one_line("out of credits") == "out of credits"


def test_the_ledger_entry_is_a_summary_not_the_whole_record(steps):
    run.run_pipeline()
    entry = run.load_ledger(config.RUN_LEDGER_FILE)["runs"][0]

    assert "warnings" not in entry
    assert "steps" not in entry
    assert entry["video_path"].endswith("reel.mp4")
    assert entry["tts_characters"] > 0


def test_the_ledger_is_trimmed_to_the_keep_limit(steps, monkeypatch, tmp_path):
    monkeypatch.setattr(config, "RUN_LEDGER_KEEP", 3)
    path = tmp_path / "small.json"
    for index in range(6):
        run.append_run({"day": f"day-{index}", "status": "ok", "spend": {}}, path)

    runs = run.load_ledger(path)["runs"]
    assert [entry["day"] for entry in runs] == ["day-3", "day-4", "day-5"]


def test_an_unreadable_ledger_is_treated_as_empty(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{not json", encoding="utf-8")
    assert run.load_ledger(path)["runs"] == []


def test_a_ledger_that_cannot_be_written_is_a_warning_not_a_failure(steps,
                                                                    monkeypatch):
    def explode(path, data):
        raise OSError("read-only filesystem")

    monkeypatch.setattr(visuals, "_write_json", explode)
    record = run.run_pipeline()

    assert record["status"] == "ok"
    assert any("ledger" in warning for warning in record["warnings"])


# ---------------------------------------------------------------------------
# The transcript
# ---------------------------------------------------------------------------

def test_the_transcript_captures_what_the_steps_printed(steps, tmp_path):
    log = tmp_path / "logs" / "run.log"

    with run.Transcript(log):
        run.run_pipeline()

    text = log.read_text(encoding="utf-8")
    assert "[1/6] topics" in text
    assert "[5/6] assemble" in text
    assert "why the ocean is salty" in text


def test_the_transcript_puts_stdout_back_afterwards(tmp_path):
    before = (sys.stdout, sys.stderr)
    with run.Transcript(tmp_path / "x.log"):
        assert sys.stdout is not before[0]
    assert (sys.stdout, sys.stderr) == before


def test_the_transcript_is_a_no_op_without_a_path():
    before = sys.stdout
    with run.Transcript(None):
        assert sys.stdout is before


def test_a_run_without_a_console_still_logs(tmp_path):
    # What pythonw.exe hands a scheduled task: sys.stdout is None.
    handle = (tmp_path / "log.txt").open("w", encoding="utf-8")
    tee = run._Tee(None, handle)
    tee.write("no console here\n")
    tee.flush()
    handle.close()

    assert "no console here" in (tmp_path / "log.txt").read_text(encoding="utf-8")
    assert tee.isatty() is False


def test_a_console_that_rejects_a_character_does_not_stop_the_run(tmp_path):
    class Cp1252Console:
        def write(self, text):
            raise UnicodeEncodeError("cp1252", text, 0, 1, "cannot encode")

        def flush(self):
            pass

    handle = (tmp_path / "log.txt").open("w", encoding="utf-8")
    run._Tee(Cp1252Console(), handle).write("an emoji topic\n")
    handle.close()

    assert "an emoji topic" in (tmp_path / "log.txt").read_text(encoding="utf-8")


def test_a_run_with_no_console_and_no_log_does_not_crash(monkeypatch, steps):
    # pythonw.exe with --no-log: nowhere to print, which must not be fatal.
    monkeypatch.setattr(sys, "stdout", None)
    monkeypatch.setattr(sys, "stderr", None)

    assert run.main(["--no-log"]) == 0
    assert isinstance(sys.stdout, run._Null)


def test_old_logs_are_pruned_and_recent_ones_kept(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    old, new = logs / "old.log", logs / "new.log"
    old.write_text("old", encoding="utf-8")
    new.write_text("new", encoding="utf-8")
    ancient = time.time() - 40 * 86400
    import os
    os.utime(old, (ancient, ancient))

    removed = run.prune_logs(30, directory=logs)

    assert removed == 1
    assert not old.exists()
    assert new.exists()


def test_pruning_is_off_when_keep_days_is_zero(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "old.log").write_text("old", encoding="utf-8")
    assert run.prune_logs(0, directory=logs) == 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_makes_a_reel_and_exits_zero(steps, capsys):
    assert run.main([]) == 0
    assert "STATUS  : OK" in capsys.readouterr().out


def test_cli_writes_a_log_by_default(steps):
    run.main([])
    logs = list((config.LOGS_DIR).glob("*.log"))
    assert len(logs) == 1
    assert "[3/6] tts" in logs[0].read_text(encoding="utf-8")


def test_cli_no_log_writes_nothing(steps):
    run.main(["--no-log"])
    assert not list(config.LOGS_DIR.glob("*.log"))


def test_cli_exits_zero_on_a_deliberate_skip(steps):
    # Non-zero would have Task Scheduler retry a run that was right to stop.
    run.main([])
    assert run.main([]) == 0


def test_cli_exits_one_on_a_failure(steps, capsys):
    steps.fail["tts"] = tts.TTSQuotaError("out of credits")
    assert run.main(["--attempts", "1"]) == 1
    assert "out of credits" in capsys.readouterr().out


def test_cli_dry_run_makes_nothing(steps, capsys):
    assert run.main(["--dry-run"]) == 0
    assert steps.calls == ["topics"]
    assert "DRY RUN" in capsys.readouterr().out


def test_cli_no_phrases_reaches_step_0(steps):
    run.main(["--dry-run", "--no-phrases"])
    assert steps.seen["find_topics_kwargs"]["niche_match"] == "terms"


def test_cli_passes_the_topic_through(steps):
    run.main(["a topic I chose", "-d", "30"])
    assert steps.seen["script_topic"] == "a topic I chose"
    assert steps.seen["script_duration"] == 30


def test_cli_marks_a_scheduled_run_in_the_ledger(steps):
    run.main(["--scheduled"])
    assert run.load_ledger(config.RUN_LEDGER_FILE)["runs"][0]["scheduled"] is True


def test_cli_json_prints_the_whole_record(steps, capsys):
    run.main(["--json"])
    record = json.loads(capsys.readouterr().out)
    assert record["status"] == "ok"
    assert record["run_id"] == "ocean-1a2b3c"


def test_cli_status_reads_the_ledger(steps, capsys):
    run.main([])
    capsys.readouterr()

    assert run.main(["--status"]) == 0
    out = capsys.readouterr().out
    assert "why the ocean is salty" in out
    assert "1 of the last 1 runs made a reel" in out


def test_cli_status_says_so_when_nothing_has_run(steps, capsys):
    assert run.main(["--status"]) == 0
    assert "no runs recorded yet" in capsys.readouterr().out


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# Reading the numbers on what is already out
# ---------------------------------------------------------------------------

def _fetching(monkeypatch, calls: list, *, recorded: int = 1,
              warnings: list[str] | None = None, boom: Exception | None = None):
    """Turn the daily read on with a provider that answers without a network."""
    monkeypatch.setattr(config, "RUN_FETCH_INSIGHTS", True)
    monkeypatch.setattr(perform, "load_provider", lambda: (lambda *a, **k: {}))

    def fetch(**kwargs):
        calls.append(kwargs)
        if boom is not None:
            raise boom
        return recorded, list(warnings or [])

    monkeypatch.setattr(perform, "fetch", fetch)


def test_a_run_reads_the_numbers_on_the_reels_already_out(steps, monkeypatch):
    calls: list = []
    _fetching(monkeypatch, calls)

    record = run.run_pipeline()

    assert len(calls) == 1, "the day's reading happened once"
    assert record["status"] == "ok"


def test_the_reading_happens_even_on_a_day_that_makes_no_reel(steps, monkeypatch):
    """The reading is about reels posted on earlier days.

    Behind the once-per-day guard, a reel posted the evening before a skipped
    day would fall out of its 24-48h window unread - which is the whole failure
    this is here to stop.
    """
    calls: list = []
    _fetching(monkeypatch, calls)
    run.run_pipeline()          # takes the day
    calls.clear()

    record = run.run_pipeline()  # the guard turns this one away

    assert record["status"] == "skipped"
    assert len(calls) == 1, "the guard skipped the reel, not the reading"


def test_a_provider_that_fails_costs_a_warning_not_the_reel(steps, monkeypatch):
    calls: list = []
    _fetching(monkeypatch, calls,
              boom=perform.PerformanceError("instagram: 429 rate limited"))

    record = run.run_pipeline()

    assert record["status"] == "ok", "the reel is the job, the numbers are not"
    assert any("insights" in w and "429" in w for w in record["warnings"])


def test_the_reading_s_own_warnings_reach_the_run(steps, monkeypatch):
    calls: list = []
    _fetching(monkeypatch, calls,
              warnings=["r1: no usable numbers in the answer"])

    record = run.run_pipeline()

    assert any("insights: r1: no usable numbers" in w
               for w in record["warnings"])


def test_no_provider_configured_reads_nothing_and_says_nothing(steps, monkeypatch):
    """Typing the numbers in by hand is a choice, not a fault - it must not
    warn on every run for the rest of time."""
    calls: list = []
    _fetching(monkeypatch, calls)
    monkeypatch.setattr(perform, "load_provider", lambda: None)

    record = run.run_pipeline()

    assert calls == []
    assert not any("insights" in w for w in record["warnings"])


def test_the_daily_read_can_be_turned_off(steps, monkeypatch):
    calls: list = []
    _fetching(monkeypatch, calls)
    monkeypatch.setattr(config, "RUN_FETCH_INSIGHTS", False)

    run.run_pipeline()

    assert calls == []


def test_a_dry_run_reads_no_numbers(steps, monkeypatch):
    """A preview makes no network calls of its own and makes none here."""
    calls: list = []
    _fetching(monkeypatch, calls)

    record = run.run_pipeline(dry_run=True)

    assert record["status"] == "skipped"
    assert calls == []
