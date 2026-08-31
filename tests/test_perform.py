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
    monkeypatch.setattr(config, "PERFORMANCE_FEEDBACK", False)
    monkeypatch.setattr(config, "PERFORMANCE_EXPLORE", "off")
    monkeypatch.setattr(config, "PERFORMANCE_MIN_SAMPLES", 3)
    monkeypatch.setattr(config, "PERFORMANCE_MIN_REELS", 4)
    monkeypatch.setattr(config, "PERFORMANCE_WINDOW_HOURS", 48)
    (tmp_path / "out").mkdir(parents=True, exist_ok=True)
    perform._CACHE.update(key=None, effects={})
    return tmp_path


def _stamp(day: int) -> str:
    """An ordered stamp, recent enough to be inside PERFORMANCE_WINDOW_HOURS.

    The tests record their numbers the instant the reel is registered, so a
    corpus dated to a fixed day in the past would be measured weeks after
    posting and thrown out of every comparison by the freshness window - which
    is the window working, not the report failing.
    """
    base = datetime.now(timezone.utc) - timedelta(hours=30)
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


def test_features_read_the_run_directory_s_manifests(paths, tmp_path):
    run_dir = tmp_path / "out" / "2026-08-24" / "ocean-1a2b3c"
    run_dir.mkdir(parents=True)
    (run_dir / "assembly.json").write_text(json.dumps({
        "duration_seconds": 52.69,
        "stats": {"shots": 16, "beats": 12, "sfx_cues": 2},
        "captions": {"backend": "elevenlabs", "loss": 0.119},
        "music": {"name": "drift"},
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
    assert features["music"] == "drift"
    assert features["tag_bands"] == {"broad": 1, "mid": 2}
    assert features["duration_band"] == "50-60s"


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

    ledger.skip("r1", "the footage was of the simile")

    assert ledger.entry("r1")["status"] == "skipped"
    assert ledger.entry("r1")["skip_reason"] == "the footage was of the simile"
    assert "metrics" not in ledger.entry("r1")


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
        ledger.record(f"r{day}", {"watch_through": 0.5})
    ledger.record("r5", {"watch_through": 1.0})

    scores = perform.ratios(ledger.rows(), "watch_through", span=6)

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
            ledger.skip(f"r{day}", "off-niche")

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
    duration, hooks, shape = perform.apply_arm({"name": "hook", "arm": "off"},
                                               duration=40)

    assert (duration, hooks, shape) == (40, False, None)
    assert perform.apply_arm({"name": "hook", "arm": "on"},
                             duration=40) == (40, None, None)


def test_the_structure_trial_turns_only_the_structure_pass_off(paths):
    duration, hooks, shape = perform.apply_arm(
        {"name": "structure", "arm": "off"}, duration=40)

    # The hook pass is left exactly where config had it: a trial varies one
    # thing, or the arms differ by two and neither can be read.
    assert (duration, hooks, shape) == (40, None, False)
    assert perform.apply_arm({"name": "structure", "arm": "on"},
                             duration=40) == (40, None, None)


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
    assert entry["metrics"] == {"views": 4200, "watch_through": 0.61}
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
