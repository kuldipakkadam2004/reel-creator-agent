"""Tests for pipeline/perform.py.

Everything here is offline and synthetic - no reel is made, no number is real.
What is under test is the bookkeeping that decides whether the ledger is honest:
that a growing account does not read as a run of good reels, that a segment of
three is not reported as a finding, that a trial actually alternates, and that
feedback stays switched off until there is something to feed back.

    python -m pytest tests/test_perform.py -v
    python tests/test_perform.py          # works without pytest installed
"""

from __future__ import annotations

import json
import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

import config  # noqa: E402
from pipeline import perform  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures and builders
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def paths(tmp_path, monkeypatch):
    """Never touch the real ledger or the real output tree."""
    monkeypatch.setattr(config, "PERFORMANCE_FILE", tmp_path / "performance.json")
    monkeypatch.setattr(config, "OUTPUT_DIR", tmp_path / "out")
    monkeypatch.setattr(config, "SCRIPTS_DIR", tmp_path / "scripts")
    monkeypatch.setattr(config, "LOGS_DIR", tmp_path / "out" / "logs")
    monkeypatch.setattr(config, "PERFORMANCE_FEEDBACK", False)
    monkeypatch.setattr(config, "PERFORMANCE_EXPLORE", "off")
    monkeypatch.setattr(config, "PERFORMANCE_MIN_SAMPLES", 3)
    monkeypatch.setattr(config, "PERFORMANCE_MIN_REELS", 4)
    monkeypatch.setattr(config, "PERFORMANCE_WINDOW_HOURS", 48)
    monkeypatch.setattr(config, "PERFORMANCE_MIN_AGE_HOURS", 24)
    monkeypatch.setattr(config, "PERFORMANCE_SETTLE_HOURS", 36)
    monkeypatch.setattr(config, "PERFORMANCE_METRIC", "watch_ratio")
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    perform._CACHE.update(key=None, effects={})
    return tmp_path


def _stamp(day: int) -> str:
    """An ordered stamp, aged to sit inside the comparable window.

    The tests record their numbers the instant the reel is registered, so a
    corpus dated to a fixed day in the past would be measured weeks after
    posting and thrown out of every comparison by the freshness window - which
    is the window working, not the report failing.

    The window has a floor as well as a ceiling now, so the base is far enough
    back that the *newest* reel in the longest corpus here (twelve of them, an
    hour apart) is still older than PERFORMANCE_MIN_AGE_HOURS. Both ends stay
    switched on for every test rather than being turned off to make the corpus
    fit: the ones that mean to be outside it say so a row at a time.
    """
    base = datetime.now(timezone.utc) - timedelta(hours=46)
    return (base + timedelta(hours=day)).isoformat(timespec="seconds")


def _features(run_id: str, *, day: int = 0, source: str = "redditrss",
              topic_components=("niche match", "explainer shape"),
              hook_components=("length",), band: str = "40-50s",
              hook_replaced: bool = True, trial: dict | None = None) -> dict:
    return {
        "features_version": perform.FEATURES_VERSION,
        "run_id": run_id,
        "day": f"2026-08-{1 + day:02d}",
        "created_at": _stamp(day),
        "topic": f"topic {run_id}",
        "topic_source": source,
        "topic_components": list(topic_components),
        "duration_band": band,
        "hook_replaced": hook_replaced,
        "hook_components": list(hook_components),
        "aligner": "elevenlabs",
        "music": "",
        "sfx_cues": 0,
        "trial": trial or {},
    }


def _ledger(*rows: dict) -> perform.PerformanceLedger:
    ledger = perform.PerformanceLedger()
    for row in rows:
        ledger.register(row)
    return ledger


# ---------------------------------------------------------------------------
# The features row
# ---------------------------------------------------------------------------

def test_features_keep_the_component_names_not_the_scores(paths):
    """The scores are this run's arithmetic under this run's weights."""
    record = {
        "run_id": "ocean-1a2b3c", "day": "2026-08-24", "run_dir": "",
        "topic": "why the ocean is salty", "topic_source": "redditrss",
        "topic_components": {"niche match": 2.0, "explainer shape": 1.2,
                             "hype in the title": 0.0},
        "duration_seconds": 41.5,
    }

    features = perform.features_for(record)

    assert features["topic_components"] == ["niche match", "explainer shape"]
    assert "2.0" not in json.dumps(features["topic_components"])


def test_a_component_that_scored_zero_did_not_fire(paths):
    record = {"run_id": "r", "topic_components": {"velocity": 0.0}}

    assert perform.features_for(record)["topic_components"] == []


def test_the_hook_components_are_the_chosen_opening_s(paths, tmp_path):
    """Not the best candidate's - the one that was actually spoken."""
    script = tmp_path / "scripts" / "salty.json"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(json.dumps({"hook": {
        "chosen": "Every raindrop carries salt to the sea.",
        "replaced": True,
        "candidates": [
            {"text": "A different opening.", "components": {"a number": 1.5}},
            {"text": "Every raindrop carries salt to the sea.",
             "components": {"length": 1.5, "strong opener": 0.8}},
        ],
    }}), encoding="utf-8")

    features = perform.features_for(
        {"run_id": "r", "script_path": str(script.with_suffix(".txt"))}
    )

    assert features["hook_components"] == ["length", "strong opener"]
    assert features["hook_replaced"] is True


def test_the_structure_components_are_the_spoken_script_s(paths, tmp_path):
    """A rejected rewrite scored too, and its components sit on disk beside the
    ones that were kept. Recording those would label the reel with a script
    nobody ever heard."""
    script = tmp_path / "scripts" / "salty.json"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(json.dumps({"structure": {
        "revised": False,
        "components": {"the ending drifts": -1.5},
        "revised_components": {"closes the loop": 2.0},
    }}), encoding="utf-8")

    features = perform.features_for(
        {"run_id": "r", "script_path": str(script.with_suffix(".txt"))}
    )

    assert features["structure_components"] == ["the ending drifts"]
    assert features["structure_revised"] is False


def test_a_rewritten_script_is_labelled_with_the_rewrite_s_components(paths, tmp_path):
    script = tmp_path / "scripts" / "salty.json"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(json.dumps({"structure": {
        "revised": True,
        "components": {"the ending drifts": -1.5},
        "revised_components": {"closes the loop": 2.0, "a turn mid-reel": 1.2},
    }}), encoding="utf-8")

    features = perform.features_for(
        {"run_id": "r", "script_path": str(script.with_suffix(".txt"))}
    )

    assert features["structure_components"] == ["closes the loop", "a turn mid-reel"]
    assert features["structure_revised"] is True


def test_a_reel_made_before_the_structure_pass_is_not_asked_about_it(paths):
    """On a version 1 row "not rewritten" and "there was no such pass" are the
    same absence, and counting them together scores reels against reels that
    never had the choice."""
    old = {**_features("r1"), "features_version": 1}
    new = _features("r2")

    assert not any(q == "structure pass" for q, _ in perform.facets(old, {}))
    assert any(q == "structure pass" for q, _ in perform.facets(new, {}))


def test_the_takeaway_components_are_the_spoken_script_s(paths, tmp_path):
    """Same rule as the structure columns, same reason: a rejected rewrite
    scored too, and recording its components would label the reel with a
    script nobody ever heard."""
    script = tmp_path / "scripts" / "salty.json"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(json.dumps({"takeaway": {
        "revised": False,
        "components": {"the viewer is in the close": 0.8},
        "revised_components": {"puts the viewer in front of it": 2.0},
    }}), encoding="utf-8")

    features = perform.features_for(
        {"run_id": "r", "script_path": str(script.with_suffix(".txt"))}
    )

    assert features["takeaway_components"] == ["the viewer is in the close"]
    assert features["takeaway_revised"] is False


def test_a_reel_handed_a_take_away_is_labelled_with_the_rewrite_s(paths, tmp_path):
    script = tmp_path / "scripts" / "salty.json"
    script.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(json.dumps({"takeaway": {
        "revised": True,
        "components": {},
        "revised_components": {"puts the viewer in front of it": 2.0,
                               "names the thing": 1.2},
    }}), encoding="utf-8")

    features = perform.features_for(
        {"run_id": "r", "script_path": str(script.with_suffix(".txt"))}
    )

    assert features["takeaway_components"] == [
        "puts the viewer in front of it", "names the thing"]
    assert features["takeaway_revised"] is True


def test_a_reel_made_before_the_take_away_pass_is_not_asked_about_it(paths):
    """Every version 3 row says "kept" because nothing existed to rewrite it.
    Counting those as reels that were offered a hand-over and did not need one
    would put the whole back catalogue in the control arm of a trial it never
    ran."""
    old = {**_features("r1"), "features_version": 3}
    new = _features("r2")

    assert not any(q == "take-away pass" for q, _ in perform.facets(old, {}))
    assert any(q == "take-away pass" for q, _ in perform.facets(new, {}))


def test_take_away_components_are_not_asked_of_an_older_row(paths):
    vocabulary = {"takeaway": ["puts the viewer in front of it"]}
    old = {**_features("r1"), "features_version": 3}
    new = {**_features("r2"),
           "takeaway_components": ["puts the viewer in front of it"]}

    assert not any(q.startswith("takeaway:")
                   for q, _ in perform.facets(old, vocabulary))
    assert ("takeaway: puts the viewer in front of it", "fired") in \
        perform.facets(new, vocabulary)


def test_features_read_the_run_directory_s_manifests(paths, tmp_path):
    run_dir = tmp_path / "out" / "2026-08-24" / "ocean-1a2b3c"
    run_dir.mkdir(parents=True)
    (run_dir / "assembly.json").write_text(json.dumps({
        "duration_seconds": 52.69,
        "stats": {"shots": 16, "beats": 12, "sfx_cues": 2},
        "captions": {"backend": "elevenlabs", "loss": 0.119},
        # The key step 5 actually writes. This fixture said "name" for a long
        # time, which is why the reader agreeing with it went unnoticed.
        "music": {"file": "drift.mp3", "credit": "\"Drift\" by x (CC BY 3.0)"},
    }), encoding="utf-8")
    (run_dir / "post.json").write_text(json.dumps({
        "mode": "llm",
        "hashtags": ["#a", "#b", "#c"],
        "hashtag_tiers": {"#a": "broad", "#b": "mid", "#c": "mid"},
    }), encoding="utf-8")

    features = perform.features_for({"run_id": "ocean-1a2b3c",
                                     "run_dir": str(run_dir)})

    assert features["shots"] == 16
    assert features["aligner"] == "elevenlabs"
    assert features["music"] == "drift.mp3"
    assert features["tag_bands"] == {"broad": 1, "mid": 2}
    assert features["duration_band"] == "50-60s"


def test_the_neutral_shot_count_reaches_the_features_row(paths, tmp_path):
    """The visuals manifest counted it from the start and nothing ever read it.

    It is the one number that separates the two reels the look pass graded 1/3
    (4 of 8 beats and 5 of 6 from the neutral shot) from the one it graded 2
    (1 of 5). Until it is on the row there is nothing for reach to test that
    against, and the cap that now bounds it is only a declared prior.
    """
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "assembly.json").write_text(json.dumps({
        "duration_seconds": 42.24,
        "stats": {"shots": 17, "beats": 8, "beats_from_neutral": 4},
    }), encoding="utf-8")

    features = perform.features_for({"run_id": "beetle-999fcdcf",
                                     "run_dir": str(run_dir)})

    assert features["beats_from_neutral"] == 4


def test_the_neutral_shot_facet_is_banded_and_version_gated():
    """A row that never counted this is not a row that counted zero."""
    row = {**_features("r1"), "beats_from_neutral": 3}
    assert ("neutral beats", "2+") in perform.facets(row, {})

    none_used = {**_features("r2"), "beats_from_neutral": 0}
    assert ("neutral beats", "0") in perform.facets(none_used, {})

    # Both absences: an older row, and a row from a run that never recorded it.
    older = {**_features("r3"), "features_version": 6, "beats_from_neutral": 3}
    unrecorded = {**_features("r4"), "beats_from_neutral": None}
    for row in (older, unrecorded):
        assert not any(name == "neutral beats"
                       for name, _ in perform.facets(row, {}))


def test_the_music_column_reads_what_step_5_actually_writes(paths, tmp_path):
    """Ties the reader to the writer instead of to a hand-typed fixture.

    The bug this pins: `Track.to_dict` writes "file", the features row read
    "name", and both sides had tests that agreed with themselves. Every reel
    ever made was recorded as having no music.
    """
    from pipeline.assemble import Track

    written = Track(path=tmp_path / "millennium-sunset.mp3",
                    credit='"Millennium sunset" by zero-project (CC BY 3.0)',
                    credit_required=True, duration=104.0).to_dict()

    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "assembly.json").write_text(
        json.dumps({"music": written}), encoding="utf-8")

    features = perform.features_for({"run_id": "r", "run_dir": str(run_dir)})
    assert features["music"] == "millennium-sunset.mp3"
    assert ("music", "on") in perform.facets(features, {})


def test_rows_written_before_the_music_fix_are_not_asked_about_music(paths):
    """They all say "" whether a bed played or not - that is a typo, not silence."""
    old = {**_features("r1"), "features_version": 2}
    new = {**_features("r2"), "music": "millennium-sunset.mp3"}

    assert not any(q == "music" for q, _ in perform.facets(old, {}))
    assert ("music", "on") in perform.facets(new, {})


def test_a_missing_manifest_costs_a_column_not_the_row(paths):
    features = perform.features_for({"run_id": "r", "run_dir": "/nope"})

    assert features["run_id"] == "r"
    assert features["shots"] is None


@pytest.mark.parametrize("seconds,band", [
    (None, "unknown"), (0, "unknown"), (29.9, "20-30s"), (30, "30-40s"),
    (52.69, "50-60s"),
])
def test_duration_bands(seconds, band):
    assert perform.duration_band(seconds) == band


def test_write_features_lands_beside_the_reel(paths, tmp_path):
    run_dir = tmp_path / "out" / "2026-08-24" / "ocean"
    run_dir.mkdir(parents=True)

    assert perform.write_features({"run_id": "ocean"}, run_dir) == []
    assert json.loads((run_dir / "features.json").read_text())["run_id"] == "ocean"


def test_write_features_without_a_directory_does_nothing(paths):
    assert perform.write_features({"run_id": "ocean", "run_dir": ""}) == []


# ---------------------------------------------------------------------------
# Reading what was typed
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("typed,expected", [
    (61, 0.61), (0.61, 0.61), (100, 1.0), (1, 1.0), (0, 0.0),
])
def test_watch_through_is_read_as_a_percentage_or_a_fraction(typed, expected):
    metrics, _ = perform.normalise_metrics({"watch_through": typed})

    assert metrics["watch_through"] == expected


def test_counts_are_integers_and_rubbish_is_a_warning():
    metrics, warnings = perform.normalise_metrics(
        {"views": 4200.7, "saves": -3, "likes": "many", "nonsense": 1}
    )

    assert metrics == {"views": 4200}
    assert "nonsense" not in metrics
    assert len(warnings) == 2


# ---------------------------------------------------------------------------
# The ledger
# ---------------------------------------------------------------------------

def test_registering_a_reel_again_never_loses_its_outcome(paths):
    ledger = _ledger(_features("r1"))
    ledger.record("r1", {"views": 100})

    ledger.register(_features("r1", topic_components=("velocity",)))

    assert ledger.entry("r1")["status"] == "posted"
    assert ledger.entry("r1")["metrics"] == {"views": 100}
    assert ledger.entry("r1")["features"]["topic_components"] == ["velocity"]


def test_numbers_can_arrive_twice_and_merge(paths):
    ledger = _ledger(_features("r1"))

    ledger.record("r1", {"views": 100})
    ledger.record("r1", {"saves": 12})

    assert ledger.entry("r1")["metrics"] == {"views": 100, "saves": 12}


def test_recording_nothing_usable_records_nothing(paths):
    ledger = _ledger(_features("r1"))

    warnings = ledger.record("r1", {"views": None, "likes": "lots"})

    assert any("no usable numbers" in w for w in warnings)
    assert ledger.entry("r1").get("status", "pending") == "pending"


def test_a_skipped_reel_keeps_its_reason_and_drops_its_numbers(paths):
    ledger = _ledger(_features("r1"))
    ledger.record("r1", {"views": 100})

    ledger.skip("r1", "footage")

    assert ledger.entry("r1")["status"] == "skipped"
    assert ledger.entry("r1")["skip_code"] == "footage"
    assert ledger.entry("r1")["skip_reason"] == \
        perform.SKIP_REASONS["footage"]
    assert "metrics" not in ledger.entry("r1")


def test_a_skip_reason_is_a_code_and_a_prefix_will_do(paths):
    ledger = _ledger(_features("r1"), _features("r2", day=1))

    assert ledger.skip("r1", "claim") == []
    assert ledger.skip("r2", "pac") == []

    assert ledger.entry("r1")["skip_code"] == "claim"
    assert ledger.entry("r2")["skip_code"] == "pacing"


def test_free_text_is_kept_as_the_note_rather_than_thrown_away(paths):
    """The sentence is still the better record. It is just not comparable."""
    ledger = _ledger(_features("r1"))

    warnings = ledger.skip("r1", "not good enough to post")

    entry = ledger.entry("r1")
    assert entry["skip_code"] == perform.UNLABELLED_SKIP
    assert entry["skip_note"] == "not good enough to post"
    assert any("not one of" in warning for warning in warnings)


def test_an_unlabelled_rejection_votes_in_neither_table(paths):
    """Fourteen rows said 'not good enough to post'. That is not evidence."""
    ledger = _corpus(8)
    for day in range(8):
        if day % 2:
            ledger.record(f"r{day}", {"watch_ratio": 0.5})
        else:
            ledger.skip(f"r{day}", "not good enough to post")

    data = perform.report(ledger=ledger, min_samples=1)

    assert data["skipped"] == 4
    assert data["binned"] == 0        # none of them carries a verdict
    assert data["unjudged"] == 4
    assert all(row["median"] == 1.0 for row in data["kept"])
    assert any("no reason on them" in warning for warning in data["warnings"])


def test_a_test_run_is_not_counted_as_a_rejected_reel(paths):
    ledger = _corpus(4)
    ledger.record("r0", {"watch_ratio": 0.5})
    ledger.skip("r1", "untested")
    ledger.skip("r2", "footage")

    data = perform.report(ledger=ledger, min_samples=1)

    assert data["binned"] == 1        # r2 alone is a verdict on a reel
    assert data["unjudged"] == 1


def test_the_binned_table_says_what_keeps_going_wrong(paths):
    ledger = _corpus(6)
    for day, code in enumerate(("footage", "footage", "footage",
                                "hook", "claim", "footage")):
        ledger.skip(f"r{day}", code)

    reasons = perform.skip_tally(ledger.rows())

    assert reasons[0]["code"] == "footage"
    assert reasons[0]["n"] == 4
    assert reasons[0]["share"] == pytest.approx(0.667, abs=0.001)
    assert reasons[0]["means"] == perform.SKIP_REASONS["footage"]


def test_a_note_travels_with_the_code(paths):
    ledger = _corpus(2)
    ledger.skip("r0", "cover")
    ledger.entry("r0")["skip_note"] = "the card gave the answer away"

    reasons = perform.skip_tally(ledger.rows())

    assert reasons[0]["notes"] == ["the card gave the answer away"]


def test_an_unknown_posting_time_is_assumed_and_says_so(paths):
    ledger = _ledger(_features("r1", day=0))

    ledger.record("r1", {"views": 100})

    entry = ledger.entry("r1")
    assert entry["posted_at_assumed"] is True
    assert entry["hours_after_post"] > 0


def test_a_given_posting_time_is_not_assumed(paths):
    ledger = _ledger(_features("r1"))

    ledger.record("r1", {"views": 100}, posted_at="2026-08-02")

    assert ledger.entry("r1")["posted_at_assumed"] is False


def test_the_ledger_survives_a_round_trip(paths):
    ledger = _ledger(_features("r1"), _features("r2", day=1))
    ledger.record("r2", {"views": 100})
    assert ledger.save() == []

    reopened = perform.PerformanceLedger()

    assert len(reopened.rows()) == 2
    assert reopened.entry("r2")["metrics"]["views"] == 100


def test_rows_come_back_oldest_first(paths):
    ledger = _ledger(_features("late", day=9), _features("early", day=1))

    assert [row["run_id"] for row in ledger.rows()] == ["early", "late"]


def test_a_corrupt_ledger_reads_as_an_empty_one(paths, tmp_path):
    (tmp_path / "performance.json").write_text("{not json", encoding="utf-8")

    assert perform.PerformanceLedger().rows() == []


def test_a_version_one_row_gets_its_ratio_from_what_it_already_held(paths, tmp_path):
    """The numbers were always there: an average watch time and a reel length."""
    (tmp_path / "performance.json").write_text(json.dumps({
        "version": 1,
        "reels": {"forest-531029b5": {
            "status": "posted",
            "run_id": "forest-531029b5",
            "features": {"run_id": "forest-531029b5",
                         "created_at": _stamp(0),
                         "spoken_seconds": 37.59},
            "metrics": {"views": 3, "avg_watch_seconds": 89.5,
                        "watch_through": 1.0},
        }},
    }), encoding="utf-8")

    ledger = perform.PerformanceLedger()

    assert ledger.migrated == 1
    assert ledger.entry("forest-531029b5")["metrics"]["watch_ratio"] == \
        pytest.approx(2.381, abs=0.001)


def test_a_skip_written_before_the_vocabulary_is_unlabelled_not_guessed(paths, tmp_path):
    (tmp_path / "performance.json").write_text(json.dumps({
        "version": 1,
        "reels": {"r1": {
            "status": "skipped",
            "run_id": "r1",
            "skip_reason": "not good enough to post",
            "features": {"run_id": "r1", "created_at": _stamp(0)},
        }},
    }), encoding="utf-8")

    entry = perform.PerformanceLedger().entry("r1")

    assert entry["skip_code"] == perform.UNLABELLED_SKIP
    assert entry["skip_note"] == "not good enough to post"


def test_the_migration_runs_once_and_then_leaves_the_file_alone(paths, tmp_path):
    ledger = _ledger(_features("r1"))
    ledger.skip("r1", "footage")
    ledger.save()

    assert perform.PerformanceLedger().migrated == 0


# ---------------------------------------------------------------------------
# Finding the reel someone meant
# ---------------------------------------------------------------------------

def _make_run_dir(tmp_path: Path, day: str, run_id: str) -> Path:
    run_dir = tmp_path / "out" / day / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / config.VIDEO_FILENAME).write_bytes(b"not really a reel")
    return run_dir


def test_a_reel_is_found_by_the_short_hash_on_the_end_of_its_id(paths, tmp_path):
    _make_run_dir(tmp_path, "2026-08-24", "why-the-ocean-is-salty-1e2c6cb6")

    assert perform.resolve_run("1e2c6cb6") == "why-the-ocean-is-salty-1e2c6cb6"


def test_a_reel_is_found_by_its_directory(paths, tmp_path):
    run_dir = _make_run_dir(tmp_path, "2026-08-24", "ocean-1a2b3c")

    assert perform.resolve_run(str(run_dir)) == "ocean-1a2b3c"


def test_latest_is_the_most_recent_reel(paths, tmp_path):
    _make_run_dir(tmp_path, "2026-08-19", "old-run")
    _make_run_dir(tmp_path, "2026-08-24", "new-run")

    assert perform.resolve_run("latest") == "new-run"


def test_an_ambiguous_reference_names_what_it_matched(paths, tmp_path):
    _make_run_dir(tmp_path, "2026-08-24", "ocean-aaa")
    _make_run_dir(tmp_path, "2026-08-24", "ocean-bbb")

    with pytest.raises(perform.PerformanceError, match="ocean-aaa"):
        perform.resolve_run("ocean")


def test_an_unknown_reference_is_an_error(paths, tmp_path):
    _make_run_dir(tmp_path, "2026-08-24", "ocean-aaa")

    with pytest.raises(perform.PerformanceError, match="no reel matches"):
        perform.resolve_run("volcano")


def test_with_no_reels_at_all_the_error_says_how_to_make_one(paths):
    with pytest.raises(perform.PerformanceError, match="run.py"):
        perform.resolve_run("anything")


# ---------------------------------------------------------------------------
# Backfilling what was made before step 9 existed
# ---------------------------------------------------------------------------

def test_backfill_adopts_reels_already_on_disk(paths, tmp_path):
    run_dir = _make_run_dir(tmp_path, "2026-08-19", "bioluminescence-v2")
    (run_dir / "assembly.json").write_text(json.dumps({
        "topic": "why deep sea creatures glow",
        "created_at": "2026-08-19T04:29:09+00:00",
        "duration_seconds": 38.2,
    }), encoding="utf-8")

    adopted, warnings = perform.backfill()

    assert (adopted, warnings) == (1, [])
    entry = perform.PerformanceLedger().entry("bioluminescence-v2")
    assert entry["features"]["topic"] == "why deep sea creatures glow"
    assert entry["features"]["backfilled"] is True
    assert entry["status"] == "pending"


def test_a_backfilled_reel_admits_it_has_no_components(paths, tmp_path):
    """Step 0's scoring is not written into the run directory."""
    _make_run_dir(tmp_path, "2026-08-19", "old-reel")

    perform.backfill()

    features = perform.PerformanceLedger().entry("old-reel")["features"]
    assert features["topic_components"] == []
    assert features["backfilled"] is True


def test_backfill_never_touches_a_reel_that_is_already_recorded(paths, tmp_path):
    _make_run_dir(tmp_path, "2026-08-19", "r1")
    ledger = _ledger(_features("r1"))
    ledger.record("r1", {"views": 4200})
    ledger.save()

    adopted, _ = perform.backfill()

    assert adopted == 0
    assert perform.PerformanceLedger().entry("r1")["metrics"] == {"views": 4200}


# ---------------------------------------------------------------------------
# Detrending - the thing that decides whether any of this means anything
# ---------------------------------------------------------------------------

def test_an_account_that_is_growing_does_not_read_as_better_reels(paths):
    """Every reel gets more views than the last, and every reel is ordinary."""
    ledger = perform.PerformanceLedger()
    for day in range(12):
        ledger.register(_features(f"r{day}", day=day))
        ledger.record(f"r{day}", {"views": 1000 + 200 * day})

    scores = perform.ratios(ledger.rows(), "views", span=6)

    # The raw numbers spread over threefold and would rank these newest-first
    # every time. Detrended, a pure trend is exactly ordinary everywhere.
    assert all(value == 1.0 for value in scores.values())
    # The first and last have no reels on one side, so they are not scored at
    # all rather than scored against a one-sided baseline.
    assert set(scores) == {f"r{day}" for day in range(1, 11)}


def test_one_reel_that_beat_its_neighbours_stands_out(paths):
    ledger = perform.PerformanceLedger()
    for day in range(11):
        ledger.register(_features(f"r{day}", day=day))
        ledger.record(f"r{day}", {"watch_ratio": 0.5})
    ledger.record("r5", {"watch_ratio": 1.0})

    scores = perform.ratios(ledger.rows(), "watch_ratio", span=6)

    assert scores["r5"] == pytest.approx(2.0)
    assert scores["r4"] == pytest.approx(1.0)


def test_a_reel_with_no_number_is_not_scored(paths):
    ledger = _ledger(_features("r1"), _features("r2", day=1),
                     _features("r3", day=2))
    ledger.record("r1", {"views": 100})
    ledger.record("r3", {"views": 100})

    scores = perform.ratios(ledger.rows(), "views", span=4)

    assert "r2" not in scores


def test_the_baseline_is_symmetric_or_it_is_nothing(paths):
    """A one-sided baseline flatters whatever is newest on a growing account."""
    values = [1.0, 2.0, 3.0, 4.0, 5.0]

    assert perform.local_baseline(values, 2, span=4) == 3.0
    assert perform.local_baseline(values, 2, span=2) == 3.0
    assert perform.local_baseline(values, 0, span=4) is None
    assert perform.local_baseline(values, 4, span=4) is None


def test_a_single_reel_has_nothing_to_be_compared_against(paths):
    assert perform.local_baseline([1.0], 0, span=6) is None


# ---------------------------------------------------------------------------
# The report
# ---------------------------------------------------------------------------

def _corpus(n: int = 12, **kwargs) -> perform.PerformanceLedger:
    ledger = perform.PerformanceLedger()
    for day in range(n):
        ledger.register(_features(f"r{day}", day=day, **kwargs))
    return ledger


def test_a_thin_segment_is_not_read_out(paths):
    ledger = perform.PerformanceLedger()
    for day in range(9):
        # One reel, alone, from a different source, in the middle where it has
        # neighbours on both sides and so does get a baseline.
        source = "wikipedia" if day == 4 else "redditrss"
        ledger.register(_features("odd" if day == 4 else f"r{day}",
                                  day=day, source=source))
        ledger.record("odd" if day == 4 else f"r{day}",
                      {"watch_through": 0.9 if day == 4 else 0.5})

    data = perform.report(ledger=ledger, min_samples=3)

    odd = [row for row in data["performance"]
           if row["question"] == "topic source" and row["answer"] == "wikipedia"]
    assert odd and odd[0]["n"] == 1
    assert odd[0]["readable"] is False


def test_a_segment_with_enough_on_both_sides_is_read_out(paths):
    ledger = perform.PerformanceLedger()
    for day in range(10):
        source = "redditrss" if day % 2 else "wikipedia"
        ledger.register(_features(f"r{day}", day=day, source=source))
        ledger.record(f"r{day}", {"watch_through": 0.8 if day % 2 else 0.4})

    data = perform.report(ledger=ledger, min_samples=3)

    reddit = next(row for row in data["performance"]
                  if row["question"] == "topic source"
                  and row["answer"] == "redditrss")
    assert reddit["readable"] is True
    assert reddit["delta"] > 0


def test_a_component_on_every_reel_is_not_a_question(paths):
    """There is nothing to compare it against."""
    ledger = _corpus(6, topic_components=("niche match",))
    for day in range(6):
        ledger.record(f"r{day}", {"watch_through": 0.5})

    data = perform.report(ledger=ledger)

    assert not any(row["question"] == "topic: niche match"
                   for row in data["performance"])


def test_the_kept_table_uses_reels_that_were_never_posted(paths):
    """The free label: it needs no API and arrives the same day."""
    ledger = perform.PerformanceLedger()
    for day in range(8):
        good = day % 2 == 0
        ledger.register(_features(f"r{day}", day=day,
                                  source="evergreen" if good else "trends"))
        if good:
            ledger.record(f"r{day}", {"watch_through": 0.5})
        else:
            ledger.skip(f"r{day}", "footage")

    data = perform.report(ledger=ledger, min_samples=3)

    evergreen = next(row for row in data["kept"]
                     if row["question"] == "topic source"
                     and row["answer"] == "evergreen")
    assert evergreen["median"] == 1.0     # every evergreen reel went out
    assert evergreen["n"] == 4
    trends = next(row for row in data["kept"]
                  if row["answer"] == "trends")
    assert trends["median"] == 0.0


def test_a_stale_measurement_is_left_out_of_the_comparison(paths):
    ledger = _corpus(8)
    for day in range(8):
        ledger.record(f"r{day}", {"watch_through": 0.5})
    ledger.entry("r3")["hours_after_post"] = 24 * 30

    data = perform.report(ledger=ledger, window_hours=48)

    assert "r3" not in data["ratios"]
    assert any("more than 48h" in warning for warning in data["warnings"])


def test_a_green_measurement_is_left_out_of_the_comparison(paths):
    """The other end of the window, and the one that was missing.

    Measured live: the first reel read through the Instagram provider entered
    the ledger 5.6 hours after it went out, with three views, and would have
    been compared against reels read a day later as though its hook had failed.
    """
    ledger = _corpus(8)
    for day in range(8):
        ledger.record(f"r{day}", {"watch_ratio": 0.5})
    ledger.entry("r3")["hours_after_post"] = 5.6

    data = perform.report(ledger=ledger, min_age_hours=24)

    assert "r3" not in data["ratios"]
    assert any("less than 24h" in warning for warning in data["warnings"])


def test_a_reel_is_re_read_until_its_numbers_settle(paths):
    """Numbers used to be taken once, and once was whenever you happened to look."""
    ledger = _corpus(3)
    ledger.record("r0", {"views": 3})
    ledger.entry("r0")["hours_after_post"] = 5.6
    ledger.record("r1", {"views": 900})
    ledger.entry("r1")["hours_after_post"] = 96.0

    unsettled = {row["run_id"] for row in ledger.unsettled(settle_hours=36)}

    assert "r0" in unsettled       # read while it was still being shown
    assert "r1" not in unsettled   # as final as this is going to make it
    assert "r2" in unsettled       # never measured at all


def test_a_reading_that_is_already_comparable_is_never_taken_again(paths):
    """The defect automating the read would otherwise have introduced.

    `report` compares on `hours_after_post`, the age of the latest reading. Two
    readings from a daily job are a day apart and the comparable window is 24h
    wide, so a reel read at 29h would be read again at 53h - comparable for
    exactly one day and then silently not.
    """
    ledger = _corpus(2)
    ledger.record("r0", {"views": 400})
    ledger.entry("r0")["hours_after_post"] = 29.2   # inside 24-48h
    ledger.record("r1", {"views": 3})
    ledger.entry("r1")["hours_after_post"] = 5.6    # too early to count

    unsettled = {row["run_id"] for row in ledger.unsettled(settle_hours=36)}

    assert "r0" not in unsettled, "asking again could only push it out"
    assert "r1" in unsettled, "still worth asking - nothing usable yet"


def test_the_settle_ceiling_still_stops_the_asking(paths):
    """The comparability guard is on top of the settle age, not instead of it."""
    ledger = _corpus(1)
    ledger.record("r0", {"views": 900})
    ledger.entry("r0")["hours_after_post"] = 96.0

    assert not ledger.unsettled(settle_hours=36)


# ---------------------------------------------------------------------------
# When a reel is due to be read
# ---------------------------------------------------------------------------

NOW = datetime(2026, 9, 11, 5, 12, tzinfo=timezone.utc)


def _went_out(hours_ago: float, read_at: float | None = None, **extra) -> dict:
    """A posted reel that went out `hours_ago`, optionally already read once."""
    row = {
        "run_id": "r0",
        "status": "posted",
        "posted_at": (NOW - timedelta(hours=hours_ago)).isoformat(),
        "hours_after_post": read_at,
    }
    row.update(extra)
    return row


def test_a_reel_too_young_to_read_says_when_it_will_be_ripe(paths):
    window = perform.read_window(_went_out(17.3, read_at=17.3), now=NOW)

    assert window["state"] == "early"
    assert window["opens_in"] == pytest.approx(6.7, abs=0.1)
    assert "ripe in 6h42m" in perform.read_due_line(_went_out(17.3, read_at=17.3), NOW)


def test_a_reel_inside_its_window_says_read_it_now(paths):
    row = _went_out(41.4, read_at=18.1)

    window = perform.read_window(row, now=NOW)

    assert window["state"] == "ripe"
    assert window["shuts_in"] == pytest.approx(6.6, abs=0.1)
    assert "READ NOW" in perform.read_due_line(row, NOW)


def test_a_reel_past_its_window_says_so_rather_than_going_quiet(paths):
    """The state worth naming: nothing can be done, and that is the point.

    A reel that was never read inside 24-48h cannot be compared with the rest
    however many times it is read afterwards, so the screen has to say that out
    loud instead of printing a due time that has already passed.
    """
    row = _went_out(70.0, read_at=5.6)

    window = perform.read_window(row, now=NOW)

    assert window["state"] == "missed"
    assert "no longer be compared" in perform.read_due_line(row, NOW)


def test_a_reel_already_read_inside_the_window_is_done(paths):
    row = _went_out(96.0, read_at=41.4)

    assert perform.read_window(row, now=NOW)["state"] == "done"


def test_an_assumed_posting_time_is_not_a_clock(paths):
    """`posted_at` falls back to when the reel was *made*, which is a guess.

    Naming a due hour off it would dress that guess up as a measurement, and
    the hour is the whole reason this exists.
    """
    row = _went_out(30.0, read_at=None, posted_at_assumed=True)

    assert perform.read_window(row, now=NOW)["state"] == "unposted"


def test_a_reel_nothing_has_posted_has_no_window_yet(paths):
    row = {"run_id": "r0", "status": "pending"}

    window = perform.read_window(row, now=NOW)

    assert window["state"] == "unposted"
    assert "not recorded as posted" in perform.read_due_line(row, NOW)


def test_the_screen_lists_a_posted_reel_that_was_read_too_early(paths, capsys):
    """The defect this whole readout exists for.

    The screen listed `pending` and `--fetch` asked about `unsettled`, and
    those are different sets: a reel posted and read at 18h is waiting for the
    reading that counts, and the screen left it out entirely. Measured on
    2026-09-11 that was a reel six hours from the end of a window nothing had
    named - and read again in time, it went from reach 7 to reach 20.
    """
    ledger = _corpus(2)
    ledger.record("r0", {"views": 11})
    ledger.entry("r0")["hours_after_post"] = 18.1   # posted, and read too early
    ledger.entry("r0")["posted_at_assumed"] = False

    perform._print_pending(ledger)

    printed = capsys.readouterr().out
    assert "r0" in printed, "the reel --fetch would ask about was not listed"


def test_re_reading_keeps_the_earlier_reading(paths):
    ledger = _corpus(1)
    ledger.record("r0", {"views": 3})
    ledger.record("r0", {"views": 412})

    entry = ledger.entry("r0")

    assert entry["metrics"]["views"] == 412
    assert [reading["metrics"]["views"]
            for reading in entry["measurements"]] == [3, 412]


def test_a_capped_watch_through_is_refused_rather_than_ranked(paths):
    """89.5s of average watch time on a 37.59s reel is not a perfect score.

    Instagram's average watch time counts replays, so a looped reel divides out
    above 1.0 and a share of the reel cannot. Capped, that reel held the
    highest watch-through in the ledger on three views.
    """
    ledger = _corpus(3)
    for day in range(3):
        ledger.record(f"r{day}", {"avg_watch_seconds": 20.0,
                                  "watch_through": 0.5, "watch_ratio": 0.5})
    ledger.record("r1", {"avg_watch_seconds": 89.5,
                         "watch_through": 1.0, "watch_ratio": 2.38})

    assert perform.saturated(ledger.entry("r1")) is True
    assert perform.metric_value(ledger.entry("r1"), "watch_through") is None
    assert perform.metric_value(ledger.entry("r1"), "watch_ratio") == 2.38

    data = perform.report(ledger=ledger, metric="watch_through")
    assert "r1" not in data["ratios"]
    assert any("watch_ratio" in warning for warning in data["warnings"])


def test_the_uncapped_ratio_still_ranks_a_looped_reel(paths):
    ledger = _corpus(11)
    for day in range(11):
        ledger.record(f"r{day}", {"watch_ratio": 0.5})
    ledger.record("r5", {"watch_ratio": 2.38})

    scores = perform.ratios(ledger.rows(), "watch_ratio", span=6)

    assert scores["r5"] == pytest.approx(4.76)


def test_a_typed_watch_through_carries_the_ratio_with_it(paths):
    metrics, _ = perform.normalise_metrics({"watch_through": 61})

    assert metrics == {"watch_through": 0.61, "watch_ratio": 0.61}


def test_a_ratio_over_one_is_not_read_as_a_percentage(paths):
    """238% and 2.38 are the same reading, and neither is 2.38%."""
    metrics, _ = perform.normalise_metrics({"watch_ratio": 2.38})

    assert metrics["watch_ratio"] == 2.38


def test_an_empty_ledger_reports_what_to_do_about_it(paths):
    data = perform.report(ledger=perform.PerformanceLedger())

    assert data["reels"] == 0
    assert any("perform.py" in warning for warning in data["warnings"])


def test_the_report_says_when_it_is_too_early_to_read(paths):
    ledger = _corpus(2)
    for day in range(2):
        ledger.record(f"r{day}", {"watch_through": 0.5})

    data = perform.report(ledger=ledger, min_samples=8)

    assert any("not a finding" in warning for warning in data["warnings"])


# ---------------------------------------------------------------------------
# Feeding back
# ---------------------------------------------------------------------------

def _measured_corpus(monkeypatch, better: str = "explainer shape"):
    """Reels carrying `better` do twice as well as the ones that do not."""
    monkeypatch.setattr(config, "PERFORMANCE_FEEDBACK", True)
    ledger = perform.PerformanceLedger()
    for day in range(12):
        has = day % 2 == 0
        ledger.register(_features(
            f"r{day}", day=day,
            topic_components=("niche match", better) if has else ("niche match",),
        ))
        ledger.record(f"r{day}", {"watch_through": 0.8 if has else 0.4})
    return ledger


def test_nothing_feeds_back_while_the_switch_is_off(paths, monkeypatch):
    ledger = _measured_corpus(monkeypatch)
    monkeypatch.setattr(config, "PERFORMANCE_FEEDBACK", False)

    assert perform.adjustment_for({"explainer shape": 1.2},
                                  ledger=ledger)["value"] == 0.0


def test_nothing_feeds_back_before_there_are_enough_reels(paths, monkeypatch):
    monkeypatch.setattr(config, "PERFORMANCE_FEEDBACK", True)
    monkeypatch.setattr(config, "PERFORMANCE_MIN_REELS", 50)
    ledger = _measured_corpus(monkeypatch)
    monkeypatch.setattr(config, "PERFORMANCE_MIN_REELS", 50)

    assert perform.adjustment_for({"explainer shape": 1.2},
                                  ledger=ledger)["value"] == 0.0


def test_a_component_that_measured_well_is_worth_something(paths, monkeypatch):
    ledger = _measured_corpus(monkeypatch)

    adjustment = perform.adjustment_for({"explainer shape": 1.2}, ledger=ledger)

    assert adjustment["value"] > 0
    assert "explainer shape" in adjustment["note"]


def test_the_adjustment_is_capped_in_both_directions(paths, monkeypatch):
    monkeypatch.setattr(config, "PERFORMANCE_MAX_ADJUST", 0.2)
    ledger = _measured_corpus(monkeypatch)

    adjustment = perform.adjustment_for({"explainer shape": 1.2}, ledger=ledger)

    assert adjustment["value"] == 0.2


def test_a_component_nobody_has_measured_is_worth_nothing(paths, monkeypatch):
    ledger = _measured_corpus(monkeypatch)

    assert perform.adjustment_for({"a component never seen": 1.0},
                                  ledger=ledger)["value"] == 0.0


# ---------------------------------------------------------------------------
# Exploration
# ---------------------------------------------------------------------------

def test_no_trial_means_no_arm(paths):
    assert perform.choose_arm("off") == {}
    assert perform.choose_arm("") == {}


def test_an_unknown_trial_is_ignored_rather_than_fatal(paths):
    assert perform.choose_arm("wingardium") == {}


def test_arms_alternate_rather_than_randomise(paths):
    """At one reel a day, randomising can hand you 19-11 by luck alone."""
    ledger = perform.PerformanceLedger()
    seen = []
    for day in range(6):
        trial = perform.choose_arm("hook", ledger=ledger)
        seen.append(trial["arm"])
        ledger.register(_features(f"r{day}", day=day, trial=trial))

    assert seen == ["on", "off", "on", "off", "on", "off"]


def test_the_hook_trial_turns_the_hook_pass_off_on_its_off_arm(paths):
    duration, hooks, shape, keep = perform.apply_arm(
        {"name": "hook", "arm": "off"}, duration=40)

    assert (duration, hooks, shape, keep) == (40, False, None, None)
    assert perform.apply_arm({"name": "hook", "arm": "on"},
                             duration=40) == (40, None, None, None)


def test_the_structure_trial_turns_only_the_structure_pass_off(paths):
    duration, hooks, shape, keep = perform.apply_arm(
        {"name": "structure", "arm": "off"}, duration=40)

    # The hook pass is left exactly where config had it: a trial varies one
    # thing, or the arms differ by two and neither can be read.
    assert (duration, hooks, shape, keep) == (40, None, False, None)
    assert perform.apply_arm({"name": "structure", "arm": "on"},
                             duration=40) == (40, None, None, None)


def test_the_takeaway_trial_turns_only_the_takeaway_pass_off(paths):
    """The trial that makes the take-away pass falsifiable.

    Its weights are a declared prior - nothing in the ledger has ever been
    saved, so there was nothing to fit them to. An arm that turns the pass off
    is the only thing that can ever say whether it was worth having.
    """
    duration, hooks, shape, keep = perform.apply_arm(
        {"name": "takeaway", "arm": "off"}, duration=40)

    assert (duration, hooks, shape, keep) == (40, None, None, False)
    assert perform.apply_arm({"name": "takeaway", "arm": "on"},
                             duration=40) == (40, None, None, None)


def test_the_takeaway_trial_alternates_like_every_other(paths):
    ledger = perform.PerformanceLedger()
    seen = [perform.choose_arm("takeaway", ledger=ledger)["arm"]
            for _ in range(2)]

    assert seen == ["on", "on"], "nothing recorded yet, so the index sticks"
    assert perform.TRIALS["takeaway"] == ("on", "off")


def test_the_duration_trial_moves_one_step_either_way(paths, monkeypatch):
    monkeypatch.setattr(config, "SCRIPT_DURATION_STEP_SECONDS", 5)
    monkeypatch.setattr(config, "SCRIPT_MIN_SECONDS", 20)
    monkeypatch.setattr(config, "SCRIPT_MAX_SECONDS", 45)

    assert perform.apply_arm({"name": "duration", "arm": "short"},
                             duration=40)[0] == 35
    assert perform.apply_arm({"name": "duration", "arm": "long"},
                             duration=40)[0] == 45


def test_a_trial_cannot_push_the_length_out_of_range(paths, monkeypatch):
    monkeypatch.setattr(config, "SCRIPT_DURATION_STEP_SECONDS", 5)
    monkeypatch.setattr(config, "SCRIPT_MAX_SECONDS", 45)
    monkeypatch.setattr(config, "SCRIPT_MIN_SECONDS", 20)

    assert perform.apply_arm({"name": "duration", "arm": "long"},
                             duration=45)[0] == 45
    assert perform.apply_arm({"name": "duration", "arm": "short"},
                             duration=20)[0] == 20


def test_the_arm_a_reel_took_is_a_question_the_report_asks(paths):
    ledger = perform.PerformanceLedger()
    for day in range(8):
        arm = "on" if day % 2 == 0 else "off"
        ledger.register(_features(f"r{day}", day=day,
                                  trial={"name": "hook", "arm": arm}))
        ledger.record(f"r{day}", {"watch_through": 0.8 if arm == "on" else 0.4})

    data = perform.report(ledger=ledger, min_samples=3)

    row = next(r for r in data["performance"]
               if r["question"] == "trial: hook" and r["answer"] == "on")
    assert row["readable"] is True
    assert row["delta"] > 0


# ---------------------------------------------------------------------------
# Providers
# ---------------------------------------------------------------------------

def test_no_provider_configured_means_no_provider(paths, monkeypatch):
    monkeypatch.setattr(config, "PERFORMANCE_PROVIDER", "")

    assert perform.load_provider() is None


def test_a_provider_that_is_not_a_dotted_path_is_rejected(paths, monkeypatch):
    monkeypatch.setattr(config, "PERFORMANCE_PROVIDER", "fetch")

    with pytest.raises(perform.PerformanceError, match="dotted path"):
        perform.load_provider()


def test_a_provider_that_cannot_be_imported_says_so(paths, monkeypatch):
    monkeypatch.setattr(config, "PERFORMANCE_PROVIDER", "nosuchmodule.fetch")

    with pytest.raises(perform.PerformanceError, match="nosuchmodule"):
        perform.load_provider()


def test_fetch_records_what_the_provider_answered(paths):
    ledger = _ledger(_features("r1"), _features("r2", day=1))

    def provider(run_ids, features):
        assert set(run_ids) == {"r1", "r2"}
        return {"r1": {"views": 4200, "watch_through": 61}}

    recorded, warnings = perform.fetch(ledger=ledger, provider=provider)

    assert (recorded, warnings) == (1, [])
    assert ledger.entry("r1")["metrics"]["watch_through"] == 0.61
    assert ledger.entry("r1")["source"] == "provider"
    assert ledger.entry("r2").get("status", "pending") == "pending"


def test_a_provider_that_raises_does_not_take_the_ledger_with_it(paths):
    ledger = _ledger(_features("r1"))

    def provider(run_ids, features):
        raise RuntimeError("token expired")

    with pytest.raises(perform.PerformanceError, match="token expired"):
        perform.fetch(ledger=ledger, provider=provider)


def test_fetch_without_a_provider_points_at_the_manual_command(paths, monkeypatch):
    monkeypatch.setattr(config, "PERFORMANCE_PROVIDER", "")

    with pytest.raises(perform.PerformanceError, match="by hand"):
        perform.fetch(ledger=_ledger(_features("r1")))


# ---------------------------------------------------------------------------
# The CLI
# ---------------------------------------------------------------------------

def test_cli_records_numbers_against_a_reel(paths, tmp_path, capsys):
    _make_run_dir(tmp_path, "2026-08-24", "ocean-1a2b3c")

    assert perform.main(["1a2b3c", "--views", "4200", "--watched", "61"]) == 0

    entry = perform.PerformanceLedger().entry("ocean-1a2b3c")
    # watch_ratio comes along uninvited: it is the same division, and a row
    # carrying only the capped figure cannot be ranked once it sits on the cap.
    assert entry["metrics"] == {"views": 4200, "watch_through": 0.61,
                                "watch_ratio": 0.61}
    assert "4,200" in capsys.readouterr().out


def test_cli_records_that_a_reel_never_went_out(paths, tmp_path, capsys):
    _make_run_dir(tmp_path, "2026-08-24", "ocean-1a2b3c")

    assert perform.main(["1a2b3c", "--skipped", "footage was wrong"]) == 0

    entry = perform.PerformanceLedger().entry("ocean-1a2b3c")
    assert entry["status"] == "skipped"
    assert "footage was wrong" in capsys.readouterr().out


def test_cli_with_no_arguments_lists_what_is_waiting(paths, tmp_path, capsys):
    _make_run_dir(tmp_path, "2026-08-24", "ocean-1a2b3c")

    assert perform.main([]) == 0

    out = capsys.readouterr().out
    assert "WAITING FOR NUMBERS" in out
    assert "ocean-1a2b3c" in out


def test_cli_reports(paths, capsys):
    ledger = _corpus(6)
    for day in range(6):
        ledger.record(f"r{day}", {"watch_through": 0.5})
    ledger.save()

    assert perform.main(["--report"]) == 0

    out = capsys.readouterr().out
    assert "HOW THEY DID" in out
    assert "KEPT OR BINNED" in out


def test_cli_json_is_parseable(paths, tmp_path, capsys):
    _make_run_dir(tmp_path, "2026-08-24", "ocean-1a2b3c")
    perform.main(["1a2b3c", "--views", "10"])
    capsys.readouterr()

    assert perform.main(["1a2b3c", "--json"]) == 0

    assert json.loads(capsys.readouterr().out)["metrics"] == {"views": 10}


def test_cli_reports_an_unknown_reel_without_a_traceback(paths, tmp_path, capsys):
    _make_run_dir(tmp_path, "2026-08-24", "ocean-1a2b3c")

    assert perform.main(["volcano", "--views", "10"]) == 1
    assert "error:" in capsys.readouterr().err


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# Saved or sent - the distribution signal, read as a rate
# ---------------------------------------------------------------------------

def test_was_kept_is_a_yes_no_not_a_count():
    assert perform.was_kept({"metrics": {"saves": 3, "shares": 0}}) == 1.0
    assert perform.was_kept({"metrics": {"saves": 0, "shares": 1}}) == 1.0
    assert perform.was_kept({"metrics": {"saves": 0, "shares": 0}}) == 0.0


def test_was_kept_tells_a_zero_from_never_measured():
    """0 saves is evidence. No saves column at all is not."""
    assert perform.was_kept({"metrics": {"saves": 0, "shares": 0}}) == 0.0
    assert perform.was_kept({"metrics": {"views": 400}}) is None
    assert perform.was_kept({}) is None


# ---------------------------------------------------------------------------
# Followed after - the one outcome that survives the reel
# ---------------------------------------------------------------------------

def test_won_a_follow_is_a_yes_no_not_a_count():
    assert perform.won_a_follow({"metrics": {"follows": 2, "profile_visits": 0}}) == 1.0
    assert perform.won_a_follow({"metrics": {"follows": 0, "profile_visits": 7}}) == 1.0
    assert perform.won_a_follow({"metrics": {"follows": 0, "profile_visits": 0}}) == 0.0


def test_won_a_follow_tells_a_zero_from_never_fetched():
    """The distinction the whole fix turns on.

    Every reel made before 2026-09-10 has no follows column, because the
    ledger held one and no fetch ever filled it. That is unmeasured. Reading
    it as 0 would print "nobody followed" over a question nothing had asked.
    """
    assert perform.won_a_follow({"metrics": {"follows": 0}}) == 0.0
    assert perform.won_a_follow({"metrics": {"views": 400, "reach": 289}}) is None
    assert perform.won_a_follow({}) is None


def test_the_report_says_when_follows_were_never_read(paths):
    """A blank column and an empty one lead to opposite decisions."""
    ledger = perform.PerformanceLedger()
    for day in range(6):
        ledger.register(_features(f"r{day}", day=day))
        ledger.record(f"r{day}", {"reach": 200, "watch_through": 0.5})

    data = perform.report(ledger=ledger, min_samples=3)

    assert "followed" in data
    assert any("nothing fetched" in warning for warning in data["warnings"])


def test_a_follows_column_of_zeros_is_not_the_same_warning(paths):
    """Read and zero is the account's state; never read is a missing fetch."""
    ledger = perform.PerformanceLedger()
    for day in range(6):
        ledger.register(_features(f"r{day}", day=day))
        ledger.record(f"r{day}", {"reach": 200, "watch_through": 0.5,
                                  "follows": 0, "profile_visits": 0})

    data = perform.report(ledger=ledger, min_samples=3)

    assert not any("nothing fetched" in warning for warning in data["warnings"])
    assert any("column of zeros" in warning and "FOLLOWED AFTER" in warning
               for warning in data["warnings"])


def test_real_engagement_counts_follows_separately_from_reels(paths):
    """A reel nothing asked for follows on is not a reel that won none."""
    rows = [
        {"status": "posted", "metrics": {"reach": 289, "likes": 4}},
        {"status": "posted",
         "metrics": {"reach": 300, "likes": 3, "follows": 2,
                     "profile_visits": 11}},
    ]

    out = perform.real_engagement(rows, own=3)

    assert out["reels"] == 2
    assert out["follow_reels"] == 1
    assert out["follows"] == 2
    assert out["profile_visits"] == 11


def test_a_metric_that_is_zero_everywhere_can_never_be_read_as_a_ratio(paths):
    """Why PERFORMANCE_METRIC was not simply switched to `saves`.

    `ratios` divides each reel by the median of its neighbours and
    `local_baseline` refuses a median of zero. Every posted reel had 0 saves,
    so ranking on saves would have produced an empty table - which reads
    exactly like "measured, no effect" and is not the same thing at all.
    """
    ledger = _corpus(12)
    for day in range(12):
        ledger.record(f"r{day}", {"saves": 0, "watch_ratio": 0.5})

    assert perform.ratios(ledger.rows(), "saves", 10) == {}
    assert perform.ratios(ledger.rows(), "watch_ratio", 10), "watch still reads"


def test_the_saved_table_reads_a_rate_where_the_ratio_table_cannot(paths):
    ledger = _corpus(12)
    for day in range(12):
        ledger.record(f"r{day}", {"saves": 1 if day % 2 else 0,
                                  "watch_ratio": 0.5})

    data = perform.report(ledger=ledger, min_samples=2)

    assert data["saved"], "a rate is defined even where a ratio is not"
    assert data["ever_saved"] == 6


def test_a_column_of_zeros_is_called_out_rather_than_read_as_no_effect(paths):
    ledger = _corpus(12)
    for day in range(12):
        ledger.record(f"r{day}", {"saves": 0, "shares": 0, "watch_ratio": 0.5})

    data = perform.report(ledger=ledger, min_samples=2)

    assert data["ever_saved"] == 0
    assert any("column of zeros" in warning for warning in data["warnings"])


def test_nothing_is_said_about_zeros_once_a_reel_has_been_saved(paths):
    ledger = _corpus(12)
    for day in range(12):
        ledger.record(f"r{day}", {"saves": 1 if day == 3 else 0,
                                  "watch_ratio": 0.5})

    data = perform.report(ledger=ledger, min_samples=2)

    assert not any("column of zeros" in warning
                   for warning in data["warnings"])


# ---------------------------------------------------------------------------
# The cover and the tags - FEATURES_VERSION 5
# ---------------------------------------------------------------------------

def test_the_cover_source_is_recorded_once_a_cover_was_written(paths, tmp_path):
    run_dir = tmp_path / "out" / "2026-09-09" / "forest-1a2b3c"
    run_dir.mkdir(parents=True)
    (run_dir / "assembly.json").write_text(json.dumps({
        "cover": {"written": True, "source": "hook_card"},
    }), encoding="utf-8")

    features = perform.features_for({"run_id": "r", "run_dir": str(run_dir)})

    assert features["cover_source"] == "hook_card"
    assert ("cover", "hook_card") in perform.facets(features, {})


def test_a_cover_that_was_never_written_is_not_a_cover_source(paths, tmp_path):
    """A run that failed before step 5 got there has no cover, which is not the
    same fact as a cover taken from the footage."""
    run_dir = tmp_path / "out" / "2026-09-09" / "forest-1a2b3c"
    run_dir.mkdir(parents=True)
    (run_dir / "assembly.json").write_text(json.dumps({
        "cover": {"written": False, "source": "hook_card"},
    }), encoding="utf-8")

    features = perform.features_for({"run_id": "r", "run_dir": str(run_dir)})

    assert features["cover_source"] == ""
    assert not any(question == "cover"
                   for question, _ in perform.facets(features, {}))


def test_a_reel_made_before_the_cover_column_is_not_asked_about_it(paths):
    old = {**_features("r1"), "features_version": 4, "cover_source": "hook_card"}
    new = {**_features("r2"), "cover_source": "hook_card"}

    assert not any(q == "cover" for q, _ in perform.facets(old, {}))
    assert ("cover", "hook_card") in perform.facets(new, {})


def test_the_tags_themselves_are_recorded_not_just_how_many(paths, tmp_path):
    """The column that makes "did #woodwideweb do anything" answerable.

    Every row up to version 5 stored the count and a tally of bands, so the
    question could not be held whatever the tags were.
    """
    run_dir = tmp_path / "out" / "2026-09-09" / "forest-1a2b3c"
    run_dir.mkdir(parents=True)
    (run_dir / "post.json").write_text(json.dumps({
        "hashtags": ["#WoodWideWeb", "#ecology", "#earthscience"],
    }), encoding="utf-8")

    features = perform.features_for({"run_id": "r", "run_dir": str(run_dir)})

    assert features["hashtags"] == 3
    assert features["hashtag_names"] == ["earthscience", "ecology",
                                         "woodwideweb"]


def test_two_spellings_of_one_tag_are_one_facet(paths, tmp_path):
    """Two spellings would be two facets that each halve their own evidence."""
    run_dir = tmp_path / "out" / "2026-09-09" / "forest-1a2b3c"
    run_dir.mkdir(parents=True)
    (run_dir / "post.json").write_text(json.dumps({
        "hashtags": ["#WoodWideWeb", "woodwideweb", " #WOODWIDEWEB ", ""],
    }), encoding="utf-8")

    features = perform.features_for({"run_id": "r", "run_dir": str(run_dir)})

    assert features["hashtag_names"] == ["woodwideweb"]


def test_a_tag_is_asked_as_a_question_like_any_other_component(paths):
    vocabulary = {"hashtag": ["woodwideweb"]}
    used = {**_features("r1"), "hashtag_names": ["woodwideweb", "ecology"]}
    unused = {**_features("r2"), "hashtag_names": ["ecology"]}

    assert ("hashtag: woodwideweb", "fired") in perform.facets(used, vocabulary)
    assert ("hashtag: woodwideweb", "absent") in perform.facets(unused,
                                                                vocabulary)


def test_a_reel_made_before_the_tags_were_written_down_is_not_asked(paths):
    """"This reel did not use #ecology" is a different fact from "nobody wrote
    the tags down"."""
    vocabulary = {"hashtag": ["ecology"]}
    old = {**_features("r1"), "features_version": 4}

    assert not any(q.startswith("hashtag:")
                   for q, _ in perform.facets(old, vocabulary))


def test_tag_effects_never_reach_step_0_s_board(paths):
    """`measured_effects` feeds topic components back onto the board. A
    hashtag is not a topic component and must not be scored as one."""
    ledger = _corpus(12)
    for day in range(12):
        row = ledger.entry(f"r{day}")["features"]
        row["hashtag_names"] = ["ecology"] if day % 2 else ["planetearth"]
        ledger.record(f"r{day}", {"watch_ratio": 1.0 if day % 2 else 0.2})

    effects = perform.measured_effects(ledger=ledger)

    assert not any(name in ("ecology", "planetearth") for name in effects)


def test_a_constant_column_is_told_apart_from_a_thin_one(paths):
    """Both are unreadable. Only one of them is fixed by making more reels."""
    ledger = perform.PerformanceLedger()
    for day in range(6):
        # `captions` is the same on every reel, so its question has one answer.
        # `topic source` has two, and is merely thin.
        ledger.register(_features(f"r{day}", day=day,
                                  source="wikipedia" if day < 2 else "redditrss"))
        ledger.record(f"r{day}", {"watch_through": 0.5})

    data = perform.report(ledger=ledger, min_samples=4)

    captions = next(row for row in data["kept"]
                    if row["question"] == "captions")
    assert captions["sides"] == 1
    assert captions["readable"] is False

    source = next(row for row in data["kept"]
                  if row["question"] == "topic source"
                  and row["answer"] == "wikipedia")
    assert source["sides"] == 2
    assert source["readable"] is False

    assert "captions" in data["one_sided"]
    assert "topic source" not in data["one_sided"]


def test_a_table_of_one_reel_calls_nothing_constant(paths):
    """One reel answers every question one way. That is not evidence."""
    rows = [{"question": "captions", "answer": "elevenlabs",
             "sides": 1, "of": 1}]
    assert perform.one_sided(rows) == []

    rows[0]["of"] = 2
    assert perform.one_sided(rows) == ["captions"]


def test_the_constant_columns_are_named_in_the_warnings(paths):
    """The one warning more patience cannot clear, so it says which."""
    ledger = perform.PerformanceLedger()
    for day in range(6):
        ledger.register(_features(f"r{day}", day=day))
        ledger.record(f"r{day}", {"watch_through": 0.5})

    data = perform.report(ledger=ledger, min_samples=4)

    flat = [w for w in data["warnings"] if "one answer on every reel" in w]
    assert len(flat) == 1
    assert "captions" in flat[0]
    assert "PERFORMANCE_EXPLORE" in flat[0]


def test_a_trial_arm_gives_a_constant_column_a_second_side(paths):
    """The point of the trial: it is what makes a column readable at all."""
    ledger = perform.PerformanceLedger()
    for day in range(6):
        arm = "on" if day % 2 else "off"
        ledger.register(_features(f"r{day}", day=day,
                                  trial={"name": "takeaway", "arm": arm}))
        ledger.record(f"r{day}", {"watch_through": 0.5})

    data = perform.report(ledger=ledger, min_samples=3)

    arms = [row for row in data["kept"]
            if row["question"] == "trial: takeaway"]
    assert {row["answer"] for row in arms} == {"on", "off"}
    assert all(row["sides"] == 2 for row in arms)
    assert "trial: takeaway" not in data["one_sided"]


# ---------------------------------------------------------------------------
# Relabelling a bin whose reason was lost
# ---------------------------------------------------------------------------

def test_the_features_row_keeps_the_log_path(paths):
    """The run directory is pruned. output/logs/ is not under it."""
    features = perform.features_for({
        "run_id": "r1",
        "run_dir": str(paths / "out" / "2026-08-24" / "r1"),
        "log_path": str(paths / "out" / "logs" / "2026-08-24-210542.log"),
    })

    assert features["log_path"].endswith("2026-08-24-210542.log")


def test_a_run_with_no_transcript_records_no_path(paths):
    assert perform.features_for({"run_id": "r1"})["log_path"] == ""


def test_unlabelled_bins_are_only_the_ones_worth_chasing(paths):
    """A test run is finished business. An unlabelled row is a lost verdict."""
    ledger = _ledger(_features("r1"), _features("r2"), _features("r3"),
                     _features("r4"))
    ledger.skip("r1", "not good enough to post")
    ledger.skip("r2", "untested")
    ledger.skip("r3", "footage")
    ledger.record("r4", {"watch_ratio": 0.5})

    assert [row["run_id"] for row in perform.unlabelled_bins(ledger.rows())] == ["r1"]


def test_the_transcript_is_found_by_the_run_id_not_the_day(paths):
    """Several runs share a day, and only one of them is this reel."""
    logs = paths / "out" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "2026-08-24-160050.log").write_text("REEL: ocean-aaaaaaaa",
                                                encoding="utf-8")
    (logs / "2026-08-24-210542.log").write_text("REEL: glacier-ec3d56f9",
                                                encoding="utf-8")

    found = perform.recover_log("glacier-ec3d56f9", {})

    assert found is not None and found.name == "2026-08-24-210542.log"


def test_the_short_id_people_type_finds_the_transcript_too(paths):
    logs = paths / "out" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    (logs / "2026-08-26-125153.log").write_text("perform.py ec3d56f9 --views",
                                                encoding="utf-8")

    assert perform.recover_log("glacier-ec3d56f9", {}) is not None


def test_a_recorded_log_path_is_used_before_any_scan(paths):
    logs = paths / "out" / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    named = logs / "2026-09-01-210819.log"
    named.write_text("nothing that mentions the id", encoding="utf-8")

    found = perform.recover_log("deserts-08022f98", {"log_path": str(named)})

    assert found == named


def test_no_transcript_left_is_not_an_error(paths):
    assert perform.recover_log("gone-56d0526c", {}) is None


def test_the_warnings_block_is_read_off_the_transcript(paths):
    log = paths / "run.log"
    log.write_text(
        "STATUS  : OK\n"
        "--------------------------------\n"
        "WARNINGS\n"
        "  - visuals: beat 6 had 0 hits\n"
        "  - assemble: 8 of 9 cuts were nudged\n"
        "--------------------------------\n",
        encoding="utf-8")

    assert perform._log_warnings(log) == [
        "visuals: beat 6 had 0 hits",
        "assemble: 8 of 9 cuts were nudged",
    ]


def test_relabel_prints_the_evidence_and_the_command(paths, capsys):
    ledger = _ledger(_features("ocean-1e2c6cb6"))
    ledger.skip("ocean-1e2c6cb6", "not good enough to post")

    perform._print_relabel(ledger)

    printed = capsys.readouterr().out
    assert "1 binned reel(s) have no reason on them" in printed
    assert "not good enough to post" in printed
    assert "--skipped <code>" in printed
    assert "footage | hook | claim" in printed


def test_relabel_says_so_when_there_is_nothing_to_do(paths, capsys):
    ledger = _ledger(_features("r1"))
    ledger.skip("r1", "footage")

    perform._print_relabel(ledger)

    assert "every binned reel carries a reason" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# How many people saw it
# ---------------------------------------------------------------------------

def test_a_reel_almost_nobody_saw_is_not_compared(paths):
    """Reach 0 carried the best watch_ratio in the file: 2.381.

    A ratio computed over one or two viewers is not a weak measurement of the
    reel, it is a measurement of who opened the app, and the report was ranking
    it against a reel that reached 289.
    """
    ledger = _corpus(6)
    for day in range(6):
        ledger.record(f"r{day}", {"watch_ratio": 0.5, "reach": 200})
    ledger.record("r3", {"watch_ratio": 2.4, "reach": 2})

    data = perform.report(ledger=ledger, min_samples=1, min_reach=30)
    everything = perform.report(ledger=ledger, min_samples=1, min_reach=0)

    assert data["unseen"] == 1
    assert any("fewer than 30 accounts" in w for w in data["warnings"])
    # One fewer row to compare, and it is the one nobody saw.
    assert data["compared"] == everything["compared"] - 1


def test_a_reel_with_no_reach_recorded_is_still_compared(paths):
    """Not knowing the reach is not knowing it was small.

    The same call the freshness window makes about a measurement with no age
    on it - and every row in the ledger before the provider was switched on
    has numbers but no reach.
    """
    ledger = _corpus(4)
    for day in range(4):
        ledger.record(f"r{day}", {"watch_ratio": 0.5})

    data = perform.report(ledger=ledger, min_samples=1, min_reach=30)

    assert data["unseen"] == 0
    assert data["compared"] == perform.report(
        ledger=ledger, min_samples=1, min_reach=0)["compared"]


def test_the_reach_floor_does_not_touch_the_kept_or_binned_table(paths):
    """Nobody has to see a reel for "it was not good enough" to be true.

    That table is a same-day verdict made before the reel went anywhere, so a
    floor about audience size has no business filtering it.
    """
    ledger = _corpus(4)
    ledger.record("r0", {"watch_ratio": 0.5, "reach": 1})
    ledger.record("r1", {"watch_ratio": 0.5, "reach": 1})
    ledger.skip("r2", "footage")
    ledger.skip("r3", "hook")

    data = perform.report(ledger=ledger, min_samples=1, min_reach=30)

    assert data["unseen"] == 2
    assert data["posted"] == 2 and data["binned"] == 2
    assert all(row["of"] == 4 for row in data["kept"])


def test_the_floor_can_be_switched_off(paths):
    ledger = _corpus(4)
    for day in range(4):
        ledger.record(f"r{day}", {"watch_ratio": 0.5, "reach": 1})

    data = perform.report(ledger=ledger, min_samples=1, min_reach=0)
    floored = perform.report(ledger=ledger, min_samples=1, min_reach=30)

    assert data["unseen"] == 0
    assert floored["unseen"] == 4, "the floor is what is being switched off"
    assert data["compared"] > floored["compared"] == 0


# ---------------------------------------------------------------------------
# Trials that reach step 5
# ---------------------------------------------------------------------------

def test_the_columns_the_report_calls_constant_can_be_trialled(paths):
    """The warning named music and motion and pointed at PERFORMANCE_EXPLORE.

    Until they were in TRIALS, that was a remedy nobody could apply: the
    setting took four names and neither of the two it was being recommended
    for was one of them.
    """
    assert {"music", "motion"} <= set(perform.TRIALS)


def test_a_music_arm_turns_the_bed_off_at_step_5(paths):
    off = perform.arm_assembly({"name": "music", "arm": "off"})
    on = perform.arm_assembly({"name": "music", "arm": "on"})

    assert off == {"music": False}, "False is 'no bed'; None is 'rotate as usual'"
    assert on == {}, "the control arm leaves config alone"


def test_a_motion_arm_freezes_the_shots(paths):
    assert perform.arm_assembly({"name": "motion", "arm": "off"}) == {"motion": "off"}
    assert perform.arm_assembly({"name": "motion", "arm": "on"}) == {}


def test_a_trial_on_something_else_changes_nothing_about_step_5(paths):
    assert perform.arm_assembly({"name": "hook", "arm": "off"}) == {}
    assert perform.arm_assembly({}) == {}


def test_the_music_and_motion_arms_alternate_like_the_rest(paths):
    ledger = _ledger(_features("r1", trial={"name": "music", "arm": "on"}))

    assert perform.choose_arm("music", ledger=ledger)["arm"] == "off"


def test_a_curiosity_arm_is_spent_on_the_board_and_nowhere_else(paths):
    """The only arm that acts at step 0, so it must not act anywhere else.

    If it also moved step 2 or step 5, its segment would be answering two
    questions and could not be read as either.
    """
    off = {"name": "curiosity", "arm": "off"}

    assert perform.arm_board(off) == {"curiosity": False},         "False is 'do not charge it'; None is 'charge it as usual'"
    assert perform.arm_board({"name": "curiosity", "arm": "on"}) == {}
    assert perform.apply_arm(off, duration=40) == (40, None, None, None)
    assert perform.arm_assembly(off) == {}


def test_a_trial_on_something_else_changes_nothing_about_the_board(paths):
    assert perform.arm_board({"name": "hook", "arm": "off"}) == {}
    assert perform.arm_board({"name": "music", "arm": "off"}) == {}
    assert perform.arm_board({}) == {}


def test_the_curiosity_arm_alternates_like_the_rest(paths):
    ledger = _ledger(_features("r1", trial={"name": "curiosity", "arm": "on"}))

    assert perform.choose_arm("curiosity", ledger=ledger)["arm"] == "off"


def test_every_board_trial_is_a_trial(paths):
    """BOARD_TRIALS names a subset, and a name in it that nothing alternates
    would be a trial the rota skips and the report still offers."""
    assert set(perform.BOARD_TRIALS) <= set(perform.TRIALS)
    for name in perform.BOARD_TRIALS:
        assert perform.arm_board({"name": name, "arm": "off"}),             f"{name} is named a board trial and arms nothing"


def test_the_report_offers_the_curiosity_arm_by_name(paths):
    """The warning that names PERFORMANCE_EXPLORE reads TRIALS, so a trial
    added without touching that sentence is still offered - which is the bug
    music and motion were added to fix."""
    ledger = _corpus(4)
    for day in range(4):
        ledger.record(f"r{day}", {"watch_ratio": 0.5, "reach": 40})

    data = perform.report(ledger=ledger, min_samples=1)
    named = [warning for warning in data["warnings"]
             if "PERFORMANCE_EXPLORE=" in warning]

    assert named and "curiosity" in named[0]


# ---------------------------------------------------------------------------
# How many people saw it, and when it went out
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("stamp, slot", [
    ("2026-09-01T17:27:24+00:00", "12-18 utc"),
    ("2026-09-08T11:48:37+00:00", "06-12 utc"),
    ("2026-09-08T00:00:00+00:00", "00-06 utc"),
    ("2026-09-08T23:59:59+00:00", "18-24 utc"),
])
def test_the_posting_hour_buckets_into_six_hour_blocks(stamp, slot):
    assert perform.posting_slot({"posted_at": stamp}) == slot


def test_an_assumed_posting_time_answers_nothing(paths):
    """`posted_at` falls back to when the reel was MADE when nobody said.

    A reel made at noon and posted in the evening would otherwise answer the
    clock question with a time nothing happened at - a wrong answer that looks
    exactly like a right one.
    """
    assert perform.posting_slot({
        "posted_at": "2026-09-01T12:00:00+00:00", "posted_at_assumed": True,
    }) == ""


def test_no_posting_time_answers_nothing(paths):
    assert perform.posting_slot({}) == ""


def test_the_clock_is_never_asked_of_kept_or_binned(paths):
    """Only posted reels have a posting time.

    Asked there, the question would split exactly the posted reels from
    exactly the binned ones and report a perfect correlation with being
    posted - the tautology that table is made of, dressed as a finding.
    """
    ledger = _corpus(6)
    for day in range(4):
        # An explicit posting time: without one it is assumed from when the
        # reel was made, and an assumed time answers no clock question at all.
        ledger.record(f"r{day}", {"watch_ratio": 0.5, "reach": 90 + day},
                      posted_at=_stamp(day))
    ledger.skip("r4", "footage")
    ledger.skip("r5", "hook")

    data = perform.report(ledger=ledger, min_samples=1)

    assert not any(row["question"] == "posted" for row in data["kept"])
    assert any(row["question"] == "posted" for row in data["reached"])


def test_the_reach_table_is_not_filtered_by_the_reach_floor(paths):
    """A reel nobody saw is a perfectly good observation OF reach.

    It is only worthless as a denominator, and filtering this table by reach
    would drop the end of the distribution the account is stuck at.
    """
    ledger = _corpus(6)
    for day in range(6):
        ledger.record(f"r{day}", {"watch_ratio": 0.5, "reach": 2})

    data = perform.report(ledger=ledger, min_samples=1, min_reach=30)

    assert data["unseen"] == 6, "none of them can be a denominator"
    assert data["reached"], "all of them are still observations of reach"
    assert data["reach_median"] == 2


def test_the_report_says_what_an_ordinary_reach_is(paths):
    ledger = _corpus(4)
    for day, reach in enumerate((3, 6, 64, 289)):
        ledger.record(f"r{day}", {"watch_ratio": 0.5, "reach": reach})

    data = perform.report(ledger=ledger, min_samples=1)

    assert data["reach_median"] == 35.0


# ---------------------------------------------------------------------------
# The reach exhibit
# ---------------------------------------------------------------------------

def test_the_exhibit_lists_reels_no_table_is_allowed_to_compare(paths):
    """Every floor that guards the ratio tables is off here, deliberately.

    A sample floor of 99 and a reach floor of 30 between them empty every
    other block in the report. This one is not a comparison, so neither
    applies: it is the list of what happened.
    """
    ledger = _corpus(3)
    ledger.record("r0", {"watch_ratio": 0.5, "reach": 2})
    ledger.record("r1", {"watch_ratio": 0.5, "reach": 289})
    ledger.record("r2", {"watch_ratio": 0.5, "reach": 9})

    data = perform.report(ledger=ledger, min_samples=99, min_reach=30)

    assert not [row for row in data["reached"] if row["readable"]]
    assert [row["reach"] for row in data["reach_exhibit"]["rows"]] == [289, 9, 2]
    assert data["reach_exhibit"]["of"] == 3


def test_the_exhibit_keeps_a_reel_the_freshness_window_threw_out(paths):
    """A week-late reading is a bad ratio and a perfectly good reach.

    The window exists so ratios compare reels of a like age. Nothing in the
    exhibit is a ratio, and dropping the oldest readings would cost most of
    the observations there are.
    """
    old = (datetime.now(timezone.utc) - timedelta(days=7)).isoformat(
        timespec="seconds")
    ledger = _corpus(2)
    ledger.record("r0", {"watch_ratio": 0.5, "reach": 64}, posted_at=old)
    ledger.record("r1", {"watch_ratio": 0.5, "reach": 6})

    data = perform.report(ledger=ledger, min_samples=1)

    assert [row["reach"] for row in data["reach_exhibit"]["rows"]] == [64, 6]
    assert any("after posting" in warning for warning in data["warnings"]),         "and it is still out of the tables"


def test_one_reel_carrying_most_of_the_reach_is_said_out_loud(paths):
    ledger = _corpus(4)
    for day, reach in enumerate((289, 9, 6, 3)):
        ledger.record(f"r{day}", {"watch_ratio": 0.5, "reach": reach})

    exhibit = perform.report(ledger=ledger, min_samples=1)["reach_exhibit"]

    assert exhibit["total"] == 307
    assert exhibit["concentration"] == round(289 / 307, 3)
    assert exhibit["concentration"] > perform._NO_MIDDLE_SHARE


def test_an_even_spread_is_not_called_out_as_one_reel(paths):
    """The sentence is arithmetic, so it has to stop being true."""
    ledger = _corpus(4)
    for day in range(4):
        ledger.record(f"r{day}", {"watch_ratio": 0.5, "reach": 50})

    exhibit = perform.report(ledger=ledger, min_samples=1)["reach_exhibit"]

    assert exhibit["concentration"] == 0.25
    assert exhibit["concentration"] < perform._NO_MIDDLE_SHARE


def test_a_reel_nobody_asked_about_is_unread_and_not_a_zero(paths):
    """Never asked is not nobody saw it - the call `won_a_follow` makes too."""
    ledger = _corpus(3)
    ledger.record("r0", {"watch_ratio": 0.5, "reach": 40})
    ledger.record("r1", {"watch_ratio": 0.5})

    exhibit = perform.report(ledger=ledger, min_samples=1)["reach_exhibit"]

    assert exhibit["unread"] == 1
    assert [row["reach"] for row in exhibit["rows"]] == [40]


def test_the_board_ordering_is_printed_beside_the_reach_ordering(paths):
    """Both orderings, and nothing inferred from the pair."""
    ledger = perform.PerformanceLedger()
    for day, (score, reach) in enumerate(((2.72, 289), (1.9, 6), (1.3, 64))):
        features = _features(f"r{day}", day=day)
        features["topic_score"] = score
        ledger.register(features)
        ledger.record(f"r{day}", {"watch_ratio": 0.5, "reach": reach})

    exhibit = perform.report(ledger=ledger, min_samples=1)["reach_exhibit"]

    assert [row["reach"] for row in exhibit["rows"]] == [289, 64, 6]
    assert [row["score"] for row in exhibit["by_score"]] == [2.72, 1.9]
    assert exhibit["by_score"][1]["reach"] == 6,         "the board's second favourite was the worst reel in the file"


def test_a_component_name_is_never_cut_in_half(paths):
    """'curiosity (contradic' reads as a component nobody recognises."""
    assert perform._charged(
        ["niche match", "curiosity (contradiction)"], width=20
    ) == "niche match, ..."
    assert perform._charged([]) == "nothing recorded"
    assert perform._charged(["averyverylongcomponentname"], width=10)         == "averyveryl", "one name too long to fit is all there is to print"


def test_the_exhibit_says_out_loud_that_it_is_not_a_finding(paths, capsys):
    ledger = _corpus(2)
    ledger.record("r0", {"watch_ratio": 0.5, "reach": 289})
    ledger.record("r1", {"watch_ratio": 0.5, "reach": 3})

    perform._print_report(perform.report(ledger=ledger, min_samples=1))
    out = capsys.readouterr().out

    assert "WHAT REACH ACTUALLY LOOKED LIKE" in out
    assert "not a finding" in out
    assert "no ordinary reel" in out


def test_nothing_posted_prints_the_exhibit_without_inventing_a_row(paths):
    ledger = _corpus(3)

    exhibit = perform.report(ledger=ledger, min_samples=1)["reach_exhibit"]

    assert exhibit["rows"] == []
    assert exhibit["concentration"] is None
    assert exhibit["total"] == 0


def test_a_batch_relabel_does_not_print_its_sentence_three_times(paths):
    """Nine rows relabelled at once carry one sentence between them.

    Printing it once says everything printing it three times does, and leaves
    room for the notes that differ - which are the ones worth reading.
    """
    ledger = _corpus(5)
    for day in range(3):
        ledger.skip(f"r{day}", "other")
        ledger.entry(f"r{day}")["skip_note"] = "reason lost"
    ledger.skip("r3", "other")
    ledger.entry("r3")["skip_note"] = "ffmpeg exited 3221225786"

    rows = perform.skip_tally(ledger.rows())

    assert rows[0]["code"] == "other" and rows[0]["n"] == 4
    assert rows[0]["notes"] == ["reason lost", "ffmpeg exited 3221225786"]



# ---------------------------------------------------------------------------
# The audience the reach figures are divided by
# ---------------------------------------------------------------------------

def test_the_ledger_keeps_one_account_block(tmp_path):
    ledger = perform.PerformanceLedger(tmp_path / "p.json")
    assert ledger.audience == {}

    ledger.set_audience({"followers": 2, "username": "reels"})
    ledger.save()

    assert perform.PerformanceLedger(tmp_path / "p.json").audience["followers"] == 2


def test_a_read_that_could_not_see_the_count_does_not_erase_the_last_one(tmp_path):
    """Not knowing today is not evidence that yesterday was wrong."""
    ledger = perform.PerformanceLedger(tmp_path / "p.json")
    ledger.set_audience({"followers": 2})
    ledger.set_audience({})
    assert ledger.audience["followers"] == 2


def test_the_account_block_survives_the_providers_own_save(tmp_path, monkeypatch):
    """The provider writes to the ledger itself while `fetch` holds a copy read
    before it ran. The per-reel fields are merged back; without the same for the
    account block the save below reverts it - which it did."""
    monkeypatch.setattr(config, "PERFORMANCE_FILE", tmp_path / "p.json")
    outer = perform.PerformanceLedger()
    outer.register({"run_id": "r1", "created_at": "2026-09-01T00:00:00+00:00"})
    outer.save()

    def provider(run_ids, features_by_run):
        # What pipeline/instagram.py does: a second ledger, written to disk.
        inner = perform.PerformanceLedger()
        inner.set_audience({"followers": 2, "username": "reels"})
        inner.save()
        return {"r1": {"views": 10, "reach": 8}}

    perform.fetch(ledger=outer, provider=provider)
    outer.save()

    assert perform.PerformanceLedger().audience["followers"] == 2


def test_the_report_says_so_when_the_audience_was_never_read(capsys):
    perform._print_audience({}, 6)
    printed = capsys.readouterr().out
    assert "not read" in printed
    assert "--fetch" in printed


def test_the_report_prints_the_audience_a_reach_is_measured_against(capsys):
    perform._print_audience(
        {"followers": 2, "username": "reels", "posts": 8,
         "read_at": "2026-09-09T14:14:37+00:00"}, 3)
    printed = capsys.readouterr().out

    assert "2 follower(s)" in printed
    assert "@reels" in printed
    # A median reel reaching 3 accounts off a following of 2 is the finding:
    # the reels are leaving the follower graph, there is just no graph.
    assert "1.50x" in printed


def test_no_ratio_is_printed_against_no_followers(capsys):
    """Dividing by zero followers is not a bigger number, it is not a number."""
    perform._print_audience({"followers": 0}, 3)
    printed = capsys.readouterr().out
    assert "0 follower(s)" in printed
    assert "x that" not in printed



# ---------------------------------------------------------------------------
# The operator's own accounts are not an audience
# ---------------------------------------------------------------------------

def _posted(reach, likes, **rest):
    metrics = {"reach": reach, "views": reach, "likes": likes,
               "comments": 0, "saves": 0, "shares": 0}
    metrics.update(rest)
    return {"status": "posted", "metrics": metrics}


def test_own_accounts_come_out_of_the_inflated_metrics(monkeypatch):
    monkeypatch.setattr(config, "PERFORMANCE_OWN_ACCOUNTS", 3)
    net = perform.net_of_own({"reach": 289, "views": 373, "likes": 4,
                              "comments": 2, "saves": 0})

    assert net["reach"] == 286
    assert net["views"] == 370
    assert net["likes"] == 1
    # Own accounts do not comment on every post, so a flat subtraction there
    # would invent comments that were never left.
    assert net["comments"] == 2


def test_nothing_is_taken_out_when_nobody_is_seeding(monkeypatch):
    monkeypatch.setattr(config, "PERFORMANCE_OWN_ACCOUNTS", 0)
    raw = {"reach": 289, "likes": 4}
    assert perform.net_of_own(raw) == raw


def test_a_stale_reading_under_the_floor_does_not_go_negative(monkeypatch):
    """`own` is a declared count. A reel read before every alt had opened it
    sits under it, which is a stale reading and not negative reach."""
    monkeypatch.setattr(config, "PERFORMANCE_OWN_ACCOUNTS", 3)
    assert perform.net_of_own({"reach": 1, "likes": 0})["reach"] == 0


def test_the_stored_row_is_never_rewritten(monkeypatch):
    """Applied when the report reads them, so lowering the setting later
    re-reads the same history rather than having overwritten it."""
    monkeypatch.setattr(config, "PERFORMANCE_OWN_ACCOUNTS", 3)
    raw = {"reach": 289, "likes": 4}
    perform.net_of_own(raw)
    assert raw == {"reach": 289, "likes": 4}


def test_a_reel_that_reached_nobody_contributes_no_engagement(monkeypatch):
    """Its likes and comments are the operator's own by definition. The whole
    row is discarded rather than netted off - which is why a comment on a reel
    with a reach of three does not survive into the total."""
    monkeypatch.setattr(config, "PERFORMANCE_OWN_ACCOUNTS", 3)
    real = perform.real_engagement([_posted(3, 3, comments=1)])

    assert real["reach"] == 0
    assert real["likes"] == 0
    assert real["comments"] == 0
    assert real["unseen"] == 1
    assert real["reels"] == 1


def test_the_one_real_like_survives_a_constant_of_three(monkeypatch):
    """The ledger as it stood on 2026-09-09. Read as a total it is twenty-five
    likes and a quiet account; read against the floor it is one like, ever."""
    monkeypatch.setattr(config, "PERFORMANCE_OWN_ACCOUNTS", 3)
    rows = [_posted(289, 4), _posted(64, 3), _posted(9, 3), _posted(6, 3),
            _posted(3, 3), _posted(3, 3), _posted(0, 3), _posted(0, 3)]

    real = perform.real_engagement(rows)

    assert real["likes"] == 1
    assert real["reach"] == 286 + 61 + 6 + 3
    assert real["unseen"] == 4
    assert real["reels"] == 8


def test_the_report_says_the_followers_are_your_own(capsys, monkeypatch):
    """A follower count that is entirely the operator's own logins is not a
    small audience, it is no audience, and the two lead to opposite calls."""
    perform._print_audience({"followers": 2, "username": "reels"}, 3, own=3,
                            real={"reels": 8, "reach": 356, "likes": 1,
                                  "comments": 0, "saves": 0, "shares": 0,
                                  "unseen": 4})
    printed = capsys.readouterr().out

    assert "your own accounts" in printed
    assert "audience here is nobody" in printed
    # And the ratio that would read as growth is not printed against them.
    assert "x that" not in printed
    assert "356 impression(s) that were not yours" in printed
    assert "1 like," in printed
    assert "0.28%" in printed


def test_the_real_lines_are_absent_when_nobody_is_seeding(capsys):
    perform._print_audience({"followers": 400, "username": "reels"}, 600,
                            own=0, real={"reels": 8, "reach": 4800})
    printed = capsys.readouterr().out

    assert "REACHED" not in printed
    assert "ACTED" not in printed
    assert "1.50x that" in printed


# ---------------------------------------------------------------------------
# What the reel actually looks like - FEATURES_VERSION 6
#
# The first columns in this row read off the finished file rather than off the
# plan that made it. Most of what is tested here is the distinction the rest of
# the version gating exists for: a reel nobody looked at is not a clean one.
# ---------------------------------------------------------------------------

def _looked_at(tmp_path, **look):
    run_dir = tmp_path / "out" / "2026-09-11" / "forest-1a2b3c"
    run_dir.mkdir(parents=True, exist_ok=True)
    payload = {"cover": {"written": True, "source": "footage"}}
    if look:
        payload["look"] = look
    (run_dir / "assembly.json").write_text(json.dumps(payload),
                                           encoding="utf-8")
    return perform.features_for({"run_id": "r", "run_dir": str(run_dir)})


def test_the_grade_of_the_finished_reel_is_recorded(paths, tmp_path):
    features = _looked_at(tmp_path, graded=True, grade=2,
                          problems=["dark", "off_topic"],
                          summary="underexposed for the first half")

    assert features["look_graded"] is True
    assert features["look_grade"] == 2
    assert features["look_problems"] == ["dark", "off_topic"]


def test_a_reel_nobody_looked_at_is_not_a_clean_one(paths, tmp_path):
    """The distinction the whole version gate exists for.

    An ungraded reel and a flawless one both carry an empty problem list. Asked
    the same questions they would score together, and the back catalogue - every
    reel made before step 5b existed - would land in the clean arm.
    """
    ungraded = _looked_at(tmp_path, graded=False, grade=None, problems=[])
    clean = _looked_at(tmp_path, graded=True, grade=3, problems=[])

    assert ungraded["look_graded"] is False
    assert clean["look_graded"] is True

    vocabulary = {"look": ["dark"]}
    assert ("looked at", "no") in perform.facets(ungraded, vocabulary)
    assert ("looked at", "yes") in perform.facets(clean, vocabulary)

    # And the ungraded one is not asked whether it was dark at all.
    assert not any(q == "look: dark" for q, _ in perform.facets(ungraded,
                                                                vocabulary))
    assert ("look: dark", "absent") in perform.facets(clean, vocabulary)


def test_an_ungraded_reel_is_not_asked_for_a_grade_it_does_not_have(paths,
                                                                    tmp_path):
    ungraded = _looked_at(tmp_path)
    assert not any(q == "look grade" for q, _ in perform.facets(ungraded, {}))


def test_a_reel_made_before_anything_looked_is_not_asked_whether_it_was(paths):
    old = {**_features("r1"), "features_version": 5, "look_graded": False}
    assert not any(q == "looked at" for q, _ in perform.facets(old, {}))


def test_the_grade_is_a_question_with_the_grade_as_its_answer(paths, tmp_path):
    features = _looked_at(tmp_path, graded=True, grade=1, problems=["blurry"])
    assert ("look grade", "1") in perform.facets(features, {})


def test_a_problem_is_asked_of_every_graded_reel_not_only_the_ones_it_hit(
        paths, tmp_path):
    vocabulary = {"look": ["caption_unreadable"]}
    hit = _looked_at(tmp_path, graded=True, grade=1,
                     problems=["caption_unreadable"])
    missed = _looked_at(tmp_path, graded=True, grade=3, problems=[])

    assert ("look: caption_unreadable", "fired") in perform.facets(hit,
                                                                   vocabulary)
    assert ("look: caption_unreadable", "absent") in perform.facets(missed,
                                                                    vocabulary)


def test_the_covers_own_grade_is_its_own_column(paths, tmp_path):
    """A different question from the reel's: worth a tap, not worth posting."""
    run_dir = tmp_path / "out" / "2026-09-11" / "forest-1a2b3c"
    run_dir.mkdir(parents=True)
    (run_dir / "assembly.json").write_text(json.dumps({
        "cover": {"written": True, "source": "footage",
                  "look": {"grade": 3, "ok": True, "why": "a glowing jellyfish"}},
        "look": {"graded": True, "grade": 2, "problems": []},
    }), encoding="utf-8")

    features = perform.features_for({"run_id": "r", "run_dir": str(run_dir)})

    assert features["cover_look_grade"] == 3
    assert features["look_grade"] == 2
    assert ("cover look grade", "3") in perform.facets(features, {})


def test_a_cover_that_was_never_written_has_no_grade_to_correlate(paths,
                                                                  tmp_path):
    run_dir = tmp_path / "out" / "2026-09-11" / "forest-1a2b3c"
    run_dir.mkdir(parents=True)
    (run_dir / "assembly.json").write_text(json.dumps({
        "cover": {"written": False, "look": {"grade": 3}},
    }), encoding="utf-8")

    features = perform.features_for({"run_id": "r", "run_dir": str(run_dir)})

    assert features["cover_look_grade"] is None
    assert not any(q == "cover look grade"
                   for q, _ in perform.facets(features, {}))


def test_a_run_with_no_assembly_at_all_still_reads(paths):
    features = perform.features_for({"run_id": "r", "run_dir": "/nope"})
    assert features["look_graded"] is False
    assert features["look_grade"] is None
    assert features["look_problems"] == []
    assert features["cover_look_grade"] is None



# ---------------------------------------------------------------------------
# Is the delta bigger than the noise?
# ---------------------------------------------------------------------------

#: The nine posted reels' reach, on 2026-09-11. Every test below uses the real
#: spread rather than an invented one: the whole question is whether a delta
#: survives *this* distribution, and a tidier one would answer an easier
#: question than the report actually faces.
_REACH = [289.0, 64.0, 20.0, 14.0, 9.0, 6.0, 3.0, 3.0, 0.0]


def _split(values_a: list[float], values_b: list[float]) -> tuple[list, callable]:
    """Rows whose `topic source` splits the two lists of readings apart."""
    rows, readings = [], {}
    for index, value in enumerate(values_a + values_b):
        run_id = f"r{index}"
        source = "redditrss" if index < len(values_a) else "wikipedia"
        rows.append({"features": _features(run_id, day=index, source=source)})
        readings[run_id] = value
    return rows, (lambda row: readings[row["features"]["run_id"]])


def test_an_interval_needs_two_readings_a_side():
    """A bucket of one resamples to itself and would report a width of zero."""
    assert perform._interval([1.0], [2.0, 3.0], statistics.median,
                             0.9, 1) == (None, None)
    assert perform._interval([1.0, 2.0], [], statistics.median,
                             0.9, 1) == (None, None)


def test_a_delta_inside_its_own_noise_is_readable_but_not_separated():
    """The defect this exists for.

    Both sides are drawn from the reach figures actually in the ledger, so
    there is no effect here to find. Eight-a-side makes the row `readable` -
    it always did - and the delta it prints is the shape of the spread. The
    row has to say so.
    """
    rows, score = _split(_REACH[:8], list(reversed(_REACH[1:])))

    out = perform._segment(rows, score, {}, min_samples=8)
    row = next(r for r in out if r["question"] == "topic source"
               and r["answer"] == "redditrss")

    assert row["readable"] is True, "eight a side; it was always printable"
    assert row["separated"] is False
    assert row["low"] < 0.0 < row["high"], "the range has to cover nought"


def test_a_difference_that_is_really_there_separates():
    rows, score = _split([9.0, 10.0, 11.0, 9.5, 10.5, 9.8, 10.2, 9.9],
                         [1.0, 2.0, 1.5, 1.2, 1.8, 1.4, 1.6, 1.3])

    row = next(r for r in perform._segment(rows, score, {}, min_samples=8)
               if r["question"] == "topic source" and r["answer"] == "redditrss")

    assert row["separated"] is True
    assert row["low"] > 0.0


def test_the_same_rows_resample_the_same_interval_twice():
    """The report is cached against the ledger's mtime and quoted in commits."""
    rows, score = _split(_REACH[:8], list(reversed(_REACH[1:])))

    first = perform._segment(rows, score, {}, min_samples=8)
    second = perform._segment(rows, score, {}, min_samples=8)

    assert [(r["low"], r["high"]) for r in first] \
        == [(r["low"], r["high"]) for r in second]


def test_a_separated_row_outranks_a_merely_printable_one():
    """Sorting on |delta| alone puts the noisiest row at the top of the table."""
    rows, score = _split(_REACH[:8], list(reversed(_REACH[1:])))
    out = [r for r in perform._segment(rows, score, {}, min_samples=8)
           if r["readable"]]

    assert out, "the fixture is meant to be readable"
    assert [r["separated"] for r in out] == sorted(
        (r["separated"] for r in out), reverse=True)


def test_noise_is_not_fed_back_onto_the_board(paths):
    """`adjustment_for` nudges step 0. Only a separated row may weigh on it."""
    ledger = perform.PerformanceLedger()
    for index, value in enumerate(_REACH[:8] + list(reversed(_REACH[1:]))):
        run_id = f"r{index}"
        components = ["niche match"] if index < 8 else ["explainer shape"]
        ledger.register(_features(run_id, day=index,
                                  topic_components=components))
        ledger.record(run_id, {"watch_ratio": value})

    effects = perform.measured_effects(ledger=ledger)

    assert "niche match" not in effects, \
        "eight a side of the ledger's own spread is not a measured effect"


def test_a_table_that_separated_nothing_says_so(capsys):
    rows, score = _split(_REACH[:8], list(reversed(_REACH[1:])))
    out = perform._segment(rows, score, {}, min_samples=8)

    perform._print_table("HOW THEY DID", out, unit="x", min_samples=8)

    printed = capsys.readouterr().out
    assert "none of the" in printed and "separated from nothing" in printed
    assert "can only see a difference past about" in printed


def test_a_table_that_separated_something_marks_it(capsys):
    rows, score = _split([9.0, 10.0, 11.0, 9.5, 10.5, 9.8, 10.2, 9.9],
                         [1.0, 2.0, 1.5, 1.2, 1.8, 1.4, 1.6, 1.3])
    out = perform._segment(rows, score, {}, min_samples=8)

    perform._print_table("HOW THEY DID", out, unit="x", min_samples=8)

    printed = capsys.readouterr().out
    assert "*" in printed
    assert "row(s) separated from nothing" in printed


def test_a_confidence_outside_the_range_does_not_index_backwards():
    """1.5 makes the tail negative, and a negative index reads the other end."""
    mine, rest = [1.0, 2.0, 3.0, 4.0], [1.0, 2.0, 3.0, 4.0]
    low, high = perform._interval(mine, rest, statistics.median, 1.5, 1)
    assert low <= high


# ---------------------------------------------------------------------------
# What was playing when it went out
#
# The `music` trial alternates whether step 5 lays a bed, which is a fact about
# the render. Instagram's licensed catalogue has no publishing API, so a
# trending track can only be attached by hand in the app after the reel leaves
# the pipeline - and a reel rendered with no bed and posted with a track on it
# is published *with* music. These pin the half the pipeline cannot observe.
# ---------------------------------------------------------------------------

def test_audio_is_recorded_without_waiting_for_the_numbers(paths):
    """The audio is known at post time; the numbers are not known for hours."""
    ledger = _ledger(_features("r0"))

    assert ledger.note_audio("r0", "added", "Blue Monday") == []

    assert ledger.entry("r0")["audio"] == {"mode": "added",
                                           "track": "Blue Monday"}


def test_an_unknown_audio_mode_is_refused_rather_than_stored(paths):
    ledger = _ledger(_features("r0"))

    with pytest.raises(perform.PerformanceError) as excinfo:
        ledger.note_audio("r0", "trending")

    assert "unknown audio mode" in str(excinfo.value)
    assert "audio" not in ledger.entry("r0")


def test_a_track_added_with_no_name_is_still_music(paths):
    """It counts as music-on; what it cannot say is which track."""
    ledger = _ledger(_features("r0"))

    warnings = ledger.note_audio("r0", "added")

    assert ledger.entry("r0")["audio"]["mode"] == "added"
    assert any("no track named" in w for w in warnings)


def test_a_row_that_was_never_told_does_not_default_to_the_bed(paths):
    """Defaulting would be right about every reel so far and silently wrong
    about the first one that is not - which is the whole point of the field."""
    assert perform._audio_posted({"status": "posted"}) == []
    assert perform._audio_posted({"audio": {"mode": "bed"}}) == [
        ("audio as posted", "bed")
    ]


def test_a_silent_arm_posted_with_a_track_is_flagged_as_music_on(paths):
    """The arm describes the render. This reel went out with music on it."""
    row = {
        "run_id": "r0",
        "features": _features("r0", trial={"name": "music", "arm": "off"}),
        "audio": {"mode": "added", "track": "Blue Monday"},
    }

    why = perform.audio_disagrees(row)

    assert "music-on however the arm is labelled" in why


def test_an_arm_that_matches_what_went_out_is_not_flagged(paths):
    row = {
        "run_id": "r0",
        "features": _features("r0", trial={"name": "music", "arm": "off"}),
        "audio": {"mode": "silent"},
    }

    assert perform.audio_disagrees(row) == ""


def test_a_trial_on_something_else_is_never_flagged_by_the_audio(paths):
    """Only the `music` trial makes a claim the audio can contradict."""
    row = {
        "run_id": "r0",
        "features": _features("r0", trial={"name": "hook", "arm": "off"}),
        "audio": {"mode": "added", "track": "Blue Monday"},
    }

    assert perform.audio_disagrees(row) == ""


def test_the_binned_table_is_never_asked_what_was_playing(paths):
    """A binned reel was never published, so the question would separate the
    posted rows from the binned ones and report the tautology."""
    assert perform._post_time_facets({"status": "skipped"}) == []
