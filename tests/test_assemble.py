"""Tests for pipeline/assemble.py.

Everything here is offline: no ffmpeg is invoked, no alignment API is called,
and no video is encoded. What is covered is the part that decides things - how
step 4's beats become shots, how words become cues, and what the ASS and the
carried-over attribution end up containing.

The encode itself is checked by running it:

    python pipeline/assemble.py -a output/audio/x.mp3 -v assets/visuals/runs/x

    python -m pytest tests/test_assemble.py -v
    python tests/test_assemble.py         # works without pytest installed
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

import config  # noqa: E402
from pipeline import assemble  # noqa: E402
from pipeline.assemble import Cue, Shot  # noqa: E402


NARRATION = 45.0

BEAT_TEXTS = [
    "Imagine living in total darkness.",
    "Sunlight vanishes after about 650 feet.",
    "So how do creatures see? They make their own light.",
]


# ---------------------------------------------------------------------------
# Fixtures - a step 4 manifest, shaped exactly like the real one
# ---------------------------------------------------------------------------

def make_asset(index: int, beat: int, kind: str = "video",
               duration: float = 12.0, **overrides) -> dict:
    asset = {
        "beat_index": beat,
        "kind": kind,
        "path": f"/nowhere/beat{beat:02d}_{index}.mp4",
        "relative_path": f"beat{beat:02d}_{index}.mp4",
        "width": 1080,
        "height": 1920,
        "duration": duration,
        "provider": "pexels",
        "provider_id": f"{1000 + index}",
        "creator": f"Creator {index}",
        "creator_url": "https://example.org/creator",
        "license": "Pexels License",
        "source_url": f"https://example.org/{index}",
        "attribution": f'"{kind}" by Creator {index} via pexels',
        "attribution_required": kind == "image",
        "role": "primary" if index % 2 == 0 else "alternate",
        "index": index,
    }
    asset.update(overrides)
    return asset


def make_manifest(assets_per_beat: int = 2, beats: int = 3,
                  narration: float = NARRATION, **overrides) -> dict:
    assets, beat_records = [], []
    span = narration / beats
    for beat in range(beats):
        indexes = []
        for slot in range(assets_per_beat):
            assets.append(make_asset(len(assets), beat))
            indexes.append(len(assets) - 1)
        beat_records.append({
            "index": beat,
            "text": BEAT_TEXTS[beat % len(BEAT_TEXTS)],
            "start": round(beat * span, 3),
            "end": round((beat + 1) * span, 3),
            "duration": round(span, 3),
            "query": "deep ocean",
            "asset_indexes": indexes,
            "status": "ok",
        })
    manifest = {
        "manifest_version": 1,
        "run_id": "test-run",
        "topic": "why deep sea creatures glow",
        "script": " ".join(BEAT_TEXTS),
        "narration_duration": narration,
        "target_duration": narration * 1.6,
        "coverage_ratio": 3.9,
        "asset_dir": "/nowhere",
        "manifest_path": "/nowhere/manifest.json",
        "beats": beat_records,
        "assets": assets,
    }
    manifest.update(overrides)
    return manifest


def probes_for(manifest: dict, ok: set[int] | None = None) -> dict[int, dict]:
    """Stand in for ffmpeg: every asset reads back at its declared length."""
    return {
        index: {
            "duration": float(asset["duration"]),
            "width": asset["width"],
            "height": asset["height"],
            "ok": ok is None or index in ok,
        }
        for index, asset in enumerate(manifest["assets"])
    }


def plan(manifest: dict, ok: set[int] | None = None, narration: float = NARRATION,
         words: list[dict] | None = None, beats: list[float] | None = None):
    return assemble.plan_timeline(manifest, narration, probes_for(manifest, ok),
                                  words, beats)


def words_from(pairs) -> list[dict]:
    return [{"text": text, "start": start, "end": end}
            for text, start, end in pairs]


# ---------------------------------------------------------------------------
# Timeline: step 4's beats are the edit
# ---------------------------------------------------------------------------

def test_shots_cover_the_narration_plus_the_tail_with_no_gaps():
    manifest = make_manifest()
    shots, _ = plan(manifest)

    assert shots[0].start == 0.0
    for earlier, later in zip(shots, shots[1:]):
        assert earlier.end == pytest.approx(later.start, abs=0.01)
    assert shots[-1].end == pytest.approx(
        NARRATION + config.VIDEO_TAIL_SECONDS, abs=0.01
    )


def test_shot_boundaries_land_on_step_4s_beat_boundaries():
    manifest = make_manifest()
    shots, _ = plan(manifest)

    starts = {round(shot.start, 2) for shot in shots}
    for beat in manifest["beats"][1:]:
        assert round(beat["start"], 2) in starts, (
            f"beat {beat['index']} does not start on a cut"
        )


def test_every_shot_uses_an_asset_step_4_gave_that_beat():
    manifest = make_manifest()
    shots, warnings = plan(manifest)

    for shot in shots:
        allowed = manifest["beats"][shot.beat_index]["asset_indexes"]
        assert shot.asset_index in allowed
    assert not any("borrow" in warning for warning in warnings)


def test_a_beat_that_re_shows_its_own_footage_says_so():
    """Step 4 now sources a clip per shot, so this is rare - and audible.

    Until this warning existed, a beat cycling back over a clip it had shown
    three seconds earlier left no trace in the run at all. The only signal was
    the look grade, which arrives after the encode has been paid for, and which
    calls it "repetitive" without saying which beat or why.
    """
    manifest = make_manifest(assets_per_beat=1, beats=3)
    shots, warnings = plan(manifest)

    repeated = [beat for beat in range(3)
                if len([s for s in shots if s.beat_index == beat])
                > len({s.asset_index for s in shots if s.beat_index == beat})]
    assert repeated, "no beat repeated a clip, so this proves nothing"
    for beat in repeated:
        assert any(f"beat {beat} was cut into" in w for w in warnings), (
            f"beat {beat} re-showed its own footage without saying so"
        )


def test_a_beat_with_a_clip_per_shot_stays_quiet():
    # Eight beats, so each is short enough that four clips covers its shots.
    # Three would make every beat a third of the narration long, and four
    # clips would not be enough for any of them.
    manifest = make_manifest(assets_per_beat=4, beats=8)
    shots, warnings = plan(manifest)
    for beat in range(8):
        here = [s for s in shots if s.beat_index == beat]
        assert len(here) == len({s.asset_index for s in here}), beat
    assert not any("same beat" in warning for warning in warnings)


def test_no_shot_outstays_the_configured_ceiling():
    manifest = make_manifest(assets_per_beat=4, beats=3)
    shots, _ = plan(manifest)

    for shot in shots[:-1]:  # the last one absorbs the rounding drift
        assert shot.duration <= config.SHOT_MAX_SECONDS + 0.01


def test_the_floor_holds_wherever_the_pacing_curve_is_not_asking_for_less():
    """SHOT_MIN_SECONDS stops the arithmetic chopping a beat into flashes. It is
    not a veto over the curve, which asks for short shots at the top on purpose,
    so it is only binding where the curve is already above it."""
    manifest = make_manifest(assets_per_beat=4, beats=3)
    shots, _ = plan(manifest)

    total = shots[-1].end
    for shot in shots[:-1]:
        # A shot is laid down at the length the curve wants where it starts.
        floor = min(config.SHOT_MIN_SECONDS,
                    assemble.shot_target_at(shot.start, total))
        assert shot.duration >= floor - 0.05

    # And nothing anywhere is shorter than the fastest the curve ever asks for.
    assert min(s.duration for s in shots) >= config.SHOT_HOOK_SECONDS - 0.05


def test_the_reel_opens_faster_than_it_settles():
    """The whole point of the curve: the first shot is materially shorter than
    the ones in the middle, where the viewer is following something."""
    manifest = make_manifest(assets_per_beat=4, beats=3)
    shots, _ = plan(manifest)

    settled = [s.duration for s in shots if 15.0 <= s.start <= 30.0]
    assert settled, "the fixture should have shots in the middle of the reel"
    assert shots[0].duration < min(settled)
    assert shots[0].duration < config.SHOT_TARGET_SECONDS


def test_a_flat_pacing_setting_puts_every_shot_on_the_same_target(monkeypatch):
    monkeypatch.setattr(config, "SHOT_PACING", "flat")
    manifest = make_manifest(assets_per_beat=4, beats=3)
    shots, _ = plan(manifest)

    for shot in shots[:-1]:
        assert shot.duration >= config.SHOT_MIN_SECONDS - 0.01
        assert shot.duration <= config.SHOT_MAX_SECONDS + 0.01
    # Every beat in the fixture is the same length, so a flat rate divides them
    # all the same way and every shot inside a beat is the same size.
    first_beat = [s.duration for s in shots if s.beat_index == 0]
    assert max(first_beat) - min(first_beat) < 0.05


def test_a_beat_cycles_its_own_assets_in_order():
    manifest = make_manifest(assets_per_beat=2, beats=3)
    shots, _ = plan(manifest)

    for beat in manifest["beats"]:
        own = beat["asset_indexes"]
        used = [s.asset_index for s in shots if s.beat_index == beat["index"]]
        assert used == [own[i % len(own)] for i in range(len(used))]


def test_a_beat_never_shows_another_beats_footage():
    manifest = make_manifest(assets_per_beat=1, beats=3)
    shots, _ = plan(manifest)

    for shot in shots:
        assert shot.asset_index in manifest["beats"][shot.beat_index]["asset_indexes"]


# ---------------------------------------------------------------------------
# Mismatches between what step 4 found and what the audio needs
# ---------------------------------------------------------------------------

def test_unreadable_assets_are_dropped_not_fatal():
    manifest = make_manifest(assets_per_beat=2, beats=3)
    shots, warnings = plan(manifest, ok={0, 1, 2, 4, 5})  # asset 3 is corrupt

    assert all(shot.asset_index != 3 for shot in shots)
    assert any("could not be read" in warning for warning in warnings)
    assert shots  # the reel is still made


def test_a_beat_with_no_footage_gives_its_time_to_a_neighbour():
    manifest = make_manifest(assets_per_beat=2, beats=3)
    shots, warnings = plan(manifest, ok={0, 1, 4, 5})  # beat 1 has nothing

    assert any("beat 1 has no usable footage" in w for w in warnings)
    assert {shot.beat_index for shot in shots} == {0, 2}
    assert shots[-1].end == pytest.approx(
        NARRATION + config.VIDEO_TAIL_SECONDS, abs=0.01
    )


def test_a_run_of_empty_beats_is_split_between_both_neighbours():
    """The deep-ocean shape: three beats fail together in the middle.

    All 12.7s of them used to land on the beat to their left, so two clips
    covered 18 of 45 seconds while the beat directly against the run - which
    had footage - was left as it was.
    """
    manifest = make_manifest(assets_per_beat=1, beats=6)
    shots, warnings = plan(manifest, ok={0, 4, 5})   # beats 1, 2 and 3 empty

    covered = {shot.beat_index for shot in shots}
    assert covered == {0, 4, 5}

    held = {index: 0.0 for index in covered}
    for shot in shots:
        held[shot.beat_index] += shot.end - shot.start

    # Beat 0 took the front half of the run and beat 4 the back half, rather
    # than beat 0 taking all of it.
    assert held[0] == pytest.approx(held[4], rel=0.25), held
    assert held[0] < NARRATION / 2, "one beat swallowed the whole run"
    for index in (1, 2, 3):
        assert any(f"beat {index} has no usable footage" in w for w in warnings)
    assert any("absorbed by beats 0 and 4" in w for w in warnings)


def test_a_trailing_empty_beat_still_goes_entirely_left():
    """There is nothing to its right to share with."""
    manifest = make_manifest(assets_per_beat=1, beats=3)
    shots, warnings = plan(manifest, ok={0, 1})      # beat 2 empty

    assert {shot.beat_index for shot in shots} == {0, 1}
    assert shots[-1].end == pytest.approx(
        NARRATION + config.VIDEO_TAIL_SECONDS, abs=0.01
    )
    assert any("absorbed by beat 1" in w for w in warnings)


def test_the_absorbed_run_leaves_no_gap_in_the_timeline():
    manifest = make_manifest(assets_per_beat=1, beats=6)
    shots, _ = plan(manifest, ok={0, 4, 5})

    assert shots[0].start == 0.0
    for earlier, later in zip(shots, shots[1:]):
        assert later.start == pytest.approx(earlier.end, abs=0.01)


def test_a_leading_empty_beat_is_absorbed_by_the_one_after_it():
    manifest = make_manifest(assets_per_beat=2, beats=3)
    shots, warnings = plan(manifest, ok={2, 3, 4, 5})  # beat 0 has nothing

    assert shots[0].start == 0.0
    assert shots[0].beat_index == 1
    assert any("beat 0 has no usable footage" in w for w in warnings)


def test_footage_shorter_than_its_slot_is_looped():
    manifest = make_manifest(assets_per_beat=1, beats=3)
    for asset in manifest["assets"]:
        asset["duration"] = 2.0
    shots, _ = plan(manifest)

    assert all(shot.looped for shot in shots)
    assert all(shot.source_start == 0.0 for shot in shots)


def test_a_clip_with_room_to_spare_cuts_in_past_its_start():
    manifest = make_manifest(assets_per_beat=2, beats=3)
    for asset in manifest["assets"]:
        asset["duration"] = 40.0
    shots, _ = plan(manifest)

    assert all(shot.source_start > 0 for shot in shots)
    assert not any(shot.looped for shot in shots)


def test_too_much_footage_is_reported_and_left_unused():
    manifest = make_manifest(assets_per_beat=6, beats=3)
    shots, warnings = plan(manifest)

    assert len({shot.asset_index for shot in shots}) < len(manifest["assets"])
    assert any("were not needed" in warning for warning in warnings)


def test_a_reused_clip_comes_back_at_a_different_in_point():
    manifest = make_manifest(assets_per_beat=1, beats=1, narration=11.0)
    manifest["assets"][0]["duration"] = 30.0
    shots, _ = plan(manifest, narration=11.0)

    assert len(shots) > 1, "an 11s beat with one long clip should still cut"
    assert all(shot.asset_index == 0 for shot in shots)
    assert len({shot.source_start for shot in shots}) == len(shots)


def test_repeating_a_still_becomes_one_longer_shot_not_two_identical_ones():
    manifest = make_manifest(assets_per_beat=1, beats=1, narration=14.0)
    manifest["assets"][0].update(kind="image", duration=0)
    shots, _ = plan(manifest, narration=14.0)

    assert len(shots) == 1
    assert shots[0].duration == pytest.approx(
        14.0 + config.VIDEO_TAIL_SECONDS, abs=0.01
    )


def test_beats_are_rescaled_when_the_audio_no_longer_matches_the_manifest():
    manifest = make_manifest(narration=45.0)
    shots, warnings = plan(manifest, narration=90.0)

    assert any("rescaled" in warning for warning in warnings)
    assert shots[-1].end == pytest.approx(90.0 + config.VIDEO_TAIL_SECONDS,
                                          abs=0.01)


def test_no_usable_asset_at_all_is_an_error_worth_reading():
    manifest = make_manifest()
    with pytest.raises(assemble.NoFootageError, match="could be opened"):
        plan(manifest, ok=set())


# ---------------------------------------------------------------------------
# Cues
# ---------------------------------------------------------------------------

def test_cues_stay_within_the_configured_word_limit():
    words = words_from([(f"word{i}", i * 0.4, i * 0.4 + 0.35) for i in range(20)])
    for cue in assemble.group_cues(words, 10.0):
        assert len(cue.words) <= config.CAPTION_MAX_WORDS


def test_a_sentence_ending_closes_the_cue():
    words = words_from([("Total", 0.0, 0.4), ("darkness.", 0.4, 0.9),
                        ("Sunlight", 0.95, 1.4)])
    cues = assemble.group_cues(words, 3.0)

    assert cues[0].text == "Total darkness."
    assert cues[1].words[0]["text"] == "Sunlight"


def test_a_long_pause_closes_the_cue():
    words = words_from([("one", 0.0, 0.3), ("two", 0.35, 0.6),
                        ("three", 4.0, 4.4)])
    cues = assemble.group_cues(words, 6.0)

    assert len(cues) == 2
    assert cues[1].words[0]["text"] == "three"


def test_cues_are_held_across_short_gaps_and_never_overlap():
    words = words_from([("one", 0.0, 0.3), ("two", 0.9, 1.2),
                        ("three", 1.5, 1.9)])
    cues = assemble.group_cues(words, 5.0)

    for earlier, later in zip(cues, cues[1:]):
        assert earlier.end <= later.start + 1e-6


def test_no_cue_outlives_the_video():
    words = words_from([("last", 9.6, 9.9)])
    cues = assemble.group_cues(words, 10.0)

    assert cues[-1].end <= 10.0


def test_whitespace_tokens_from_the_aligner_are_dropped():
    raw = [{"text": "one", "start": 0.0, "end": 0.3},
           {"text": " ", "start": 0.3, "end": 0.4},
           {"text": "two", "start": 0.4, "end": 0.7}]
    assert [w["text"] for w in assemble._clean_words(raw)] == ["one", "two"]


# ---------------------------------------------------------------------------
# Alignment backends
# ---------------------------------------------------------------------------

def test_estimate_spreads_words_across_step_4s_beat_windows():
    manifest = make_manifest()
    words, loss = assemble._align_estimate(manifest, NARRATION)

    assert words[0]["start"] == pytest.approx(0.0, abs=0.01)
    assert words[-1]["end"] == pytest.approx(NARRATION, abs=0.05)
    assert loss == 0.0
    for earlier, later in zip(words, words[1:]):
        assert earlier["start"] <= later["start"]


def test_a_failing_backend_falls_through_to_the_next(monkeypatch):
    manifest = make_manifest()
    monkeypatch.setattr(assemble, "_align_whisper",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("no model")))

    words, backend, _loss, warnings = assemble.align_words(
        Path("nowhere.mp3"), manifest, NARRATION, "whisper"
    )
    assert backend == "estimate" and words
    assert any("whisper failed" in warning for warning in warnings)


def test_an_unknown_backend_name_is_rejected():
    with pytest.raises(assemble.AssemblyError, match="unknown alignment"):
        assemble.align_words(Path("nowhere.mp3"), make_manifest(), NARRATION,
                             "telepathy")


# ---------------------------------------------------------------------------
# ASS
# ---------------------------------------------------------------------------

def ass_events(text: str) -> list[str]:
    body = text.split("[Events]", 1)[1].splitlines()
    return [line for line in body if line.startswith("Dialogue:")]


def test_highlight_style_emits_one_line_per_word_with_a_colour_override():
    cue = Cue(words_from([("one", 0.0, 0.3), ("two", 0.3, 0.6),
                          ("three", 0.6, 0.9)]), 0.0, 0.9)
    events = ass_events(assemble.build_ass([cue], "Arial Black", None))

    assert len(events) == 3
    for line in events:
        assert line.count("{\\c&H") == 2, "one word tinted, then set back"
    assert "ONE TWO THREE" in events[0].replace("{\\c&HFFE14D&}", "") \
                                       .replace("{\\c&HFFFFFF&}", "")


def test_plain_style_emits_one_line_per_cue(monkeypatch):
    monkeypatch.setattr(config, "CAPTION_STYLE", "plain")
    cue = Cue(words_from([("one", 0.0, 0.3), ("two", 0.3, 0.6)]), 0.0, 0.6)
    events = ass_events(assemble.build_ass([cue], "Arial Black", None))

    assert len(events) == 1
    assert "{\\c&H" not in events[0]


def test_ass_timestamps_span_exactly_the_cue():
    cue = Cue(words_from([("a", 1.0, 1.2), ("b", 1.2, 1.5)]), 1.0, 2.0)
    events = ass_events(assemble.build_ass([cue], "Arial Black", None))

    assert events[0].split(",")[1] == "0:00:01.00", "the cue opens on its first word"
    assert events[-1].split(",")[2] == "0:00:02.00", "and runs to the cue end"
    for earlier, later in zip(events, events[1:]):
        assert earlier.split(",")[2] == later.split(",")[1], "no gap between words"


def test_a_word_too_fast_to_show_still_gets_a_frame():
    cue = Cue(words_from([("a", 0.0, 0.001), ("b", 0.001, 0.002),
                          ("c", 0.002, 0.5)]), 0.0, 0.5)
    events = ass_events(assemble.build_ass([cue], "Arial Black", None))

    starts = [line.split(",")[1] for line in events]
    assert len(set(starts)) == len(starts), "two words share a start time"


def test_uppercase_is_applied_when_configured(monkeypatch):
    monkeypatch.setattr(config, "CAPTION_UPPERCASE", False)
    cue = Cue(words_from([("quiet", 0.0, 0.4)]), 0.0, 0.4)
    assert "quiet" in assemble.build_ass([cue], "Arial Black", None)


def test_braces_and_backslashes_cannot_escape_into_ass_markup():
    cue = Cue(words_from([("{\\an8}drop", 0.0, 0.4)]), 0.0, 0.4)
    events = ass_events(assemble.build_ass([cue], "Arial Black", None))

    assert "\\an8" not in events[0]
    assert "(" in events[0]


#: A font-size override and nothing else - "\\fs" on its own also matches
#: the "\\fscx" of the caption pop, which is a different tag.
FONT_SIZE_TAG = re.compile(r"\\fs\d")


def test_a_cue_too_wide_for_the_frame_is_scaled_down():
    wide = Cue(words_from([("SUPERCALIFRAGILISTICEXPIALIDOCIOUS", 0.0, 1.0)]),
               0.0, 1.0)
    assert FONT_SIZE_TAG.search(assemble.build_ass([wide], "Arial Black", None))

    narrow = Cue(words_from([("SHORT", 0.0, 1.0)]), 0.0, 1.0)
    assert not FONT_SIZE_TAG.search(
        assemble.build_ass([narrow], "Arial Black", None))


def test_colours_convert_to_ass_bgr_order():
    assert assemble._ass_colour("#FF0000") == "&H000000FF"   # red
    assert assemble._ass_colour("#00FF00") == "&H0000FF00"   # green
    assert assemble._ass_colour("#0000FF") == "&H00FF0000"   # blue
    assert assemble._ass_colour("#FFF") == "&H00FFFFFF"


def test_a_bad_colour_is_rejected_by_name():
    with pytest.raises(assemble.AssemblyError, match="not a #RRGGBB"):
        assemble._ass_colour("cornflower")


# ---------------------------------------------------------------------------
# Caption pop: a cue arrives, and a payoff word is made
# ---------------------------------------------------------------------------

def pop_of(line: str) -> str:
    """The override block a dialogue line opens with, or "" if it has none."""
    body = line.split(",,0,0,0,,", 1)[1]
    return body.split("}", 1)[0] + "}" if body.startswith("{") else ""


def test_a_cue_lands_oversized_and_settles_to_full_size():
    cue = Cue(words_from([("salt", 0.0, 0.4), ("water", 0.4, 0.9)]), 0.0, 0.9)
    first = ass_events(assemble.build_ass([cue], "Arial Black", None))[0]

    assert f"\\fscx{config.CAPTION_POP_SCALE}" in pop_of(first)
    assert f"\\t(0,{config.CAPTION_POP_MS},\\fscx100\\fscy100)" in pop_of(first)


def test_only_the_opening_word_of_a_cue_pops():
    """The pop marks the cue arriving. Popping on every word is an animation."""
    cue = Cue(words_from([("deep", 0.0, 0.4), ("dark", 0.4, 0.8),
                          ("cold", 0.8, 1.2)]), 0.0, 1.2)
    events = ass_events(assemble.build_ass([cue], "Arial Black", None))

    assert "\\fscx" in pop_of(events[0])
    assert all("\\fscx" not in pop_of(line) for line in events[1:])


def test_the_payoff_word_gets_its_own_colour_and_a_harder_pop():
    cue = Cue(words_from([("it", 0.0, 0.4), ("is", 0.6, 1.2),
                          ("fifty", 1.4, 1.9)]), 0.0, 1.9)
    events = ass_events(assemble.build_ass([cue], "Arial Black", None))
    payoff_tint = assemble._ass_inline_colour(config.CAPTION_PAYOFF_COLOR)
    highlight_tint = assemble._ass_inline_colour(config.CAPTION_HIGHLIGHT_COLOR)

    assert highlight_tint in events[0], "a connective keeps the ordinary tint"
    assert payoff_tint in events[2], "a spelled-out number is a payoff"
    assert f"\\fscx{config.CAPTION_PAYOFF_SCALE}" in pop_of(events[2])
    assert f"\\fscx{config.CAPTION_POP_SCALE}" in pop_of(events[0])


def test_a_cue_that_already_fills_the_line_is_not_popped():
    """Scaling happens after the line breaks are decided, so popping a cue with
    no room left wraps it for 90ms and snaps it back - a rendering fault, not
    emphasis. Measured on the salt reel: "50 QUADRILLION TONS" went to three
    rows at 1.30s and back to one at 1.40s."""
    full = Cue(words_from([("about", 0.0, 0.4), ("ninety-five", 0.6, 1.2),
                           ("percent", 1.4, 1.9)]), 0.0, 1.9)
    events = ass_events(assemble.build_ass([full], "Arial Black", None))

    assert all("\\fscx" not in pop_of(line) for line in events)


def test_the_cap_is_taken_from_the_room_the_cue_actually_has():
    """A short cue keeps the full pop; a wide one gets whatever is left."""
    assert assemble._fit_pop_scale("NO", None, 96, 112) == 112
    assert assemble._fit_pop_scale("W" * 40, None, 96, 112) == 100
    # ...and never below 100, which would shrink the text instead of popping it.
    assert assemble._fit_pop_scale("W" * 200, None, 96, 124) == 100


def test_a_payoff_pop_never_interrupts_the_one_before_it(monkeypatch):
    """"Ninety-five percent" is two payoffs a breath apart, and two accents
    inside one settle read as a wobble rather than as two words being made."""
    monkeypatch.setattr(config, "CAPTION_POP_MS", 90)
    cue = Cue(words_from([("ninety-five", 0.0, 0.05), ("percent", 0.05, 0.4),
                          ("unexplored", 0.4, 1.1)]), 0.0, 1.1)
    events = ass_events(assemble.build_ass([cue], "Arial Black", None))
    payoff_tint = assemble._ass_inline_colour(config.CAPTION_PAYOFF_COLOR)

    assert "\\fscx" in pop_of(events[0])
    assert "\\fscx" not in pop_of(events[1]), "0.05s in, the first pop is still running"
    assert payoff_tint in events[1], "but it is still the payoff colour"


def test_digits_are_a_payoff_and_ordinary_words_are_not():
    """Step 2 is told to write numbers as they are said, so the captions carry
    words far more often than digits - but both have to count."""
    for word in ("95%", "200", "ninety-five", "Quadrillion", "percent."):
        assert assemble._is_payoff_word(word), word
    for word in ("ocean", "the", "half", "deep-sea", ""):
        assert not assemble._is_payoff_word(word), word


def test_plain_style_pops_the_cue_and_tints_nothing(monkeypatch):
    """One colour for the whole cue is what "plain" means; a payoff tint in it
    would make the setting a lie."""
    monkeypatch.setattr(config, "CAPTION_STYLE", "plain")
    cue = Cue(words_from([("fifty", 0.0, 0.4), ("tons", 0.4, 0.9)]), 0.0, 0.9)
    events = ass_events(assemble.build_ass([cue], "Arial Black", None))

    assert len(events) == 1
    assert "\\fscx" in pop_of(events[0])
    assert "{\\c&H" not in events[0]


def test_pop_off_leaves_no_transform_in_the_file(monkeypatch):
    monkeypatch.setattr(config, "CAPTION_POP", False)
    cue = Cue(words_from([("fifty", 0.0, 0.4), ("tons", 0.4, 0.9)]), 0.0, 0.9)
    body = assemble.build_ass([cue], "Arial Black", None)

    assert "\\fscx" not in body and "\\t(" not in body
    assert assemble._ass_inline_colour(config.CAPTION_PAYOFF_COLOR) in body,         "the payoff colour is a separate look, disabled separately"


def test_srt_is_written_in_srt_time_format():
    cue = Cue(words_from([("hello", 1.5, 2.25)]), 1.5, 2.25)
    srt = assemble.build_srt([cue])

    assert "00:00:01,500 --> 00:00:02,250" in srt
    assert "HELLO" in srt


# ---------------------------------------------------------------------------
# ffmpeg arguments
# ---------------------------------------------------------------------------

def shot(**overrides) -> Shot:
    fields = {"index": 0, "beat_index": 0, "asset_index": 0,
              "path": Path("/nowhere/clip.mp4"), "kind": "video",
              "start": 0.0, "duration": 3.0}
    fields.update(overrides)
    return Shot(**fields)


def test_a_short_clip_is_looped_rather_than_seeked():
    args = assemble._shot_inputs(shot(looped=True))
    assert "-stream_loop" in args and "-ss" not in args


# --- what the timeline remembers about its footage -------------------------

def test_a_shot_carries_the_provider_that_supplied_it():
    """Without this the assembly cannot say where a reel's pictures came from."""
    manifest = {"assets": [{"provider": "pexels", "provider_key": "pexels:1",
                            "score": 14.5, "relevance": 0.5,
                            "width": 1080, "height": 1920}]}

    carried = assemble._shot_provenance(manifest, shot(asset_index=0))

    assert carried["provider"] == "pexels"
    assert carried["score"] == 14.5
    assert carried["width"] == 1080


def test_a_field_step_four_never_wrote_is_absent_rather_than_null():
    """"Not recorded" and "recorded as nothing" are different answers."""
    manifest = {"assets": [{"provider": "openverse"}]}

    carried = assemble._shot_provenance(manifest, shot(asset_index=0))

    assert carried == {"provider": "openverse"}


def test_a_shot_pointing_past_the_manifest_costs_nothing():
    """A stale index must not take the whole assembly down with it."""
    assert assemble._shot_provenance({"assets": []}, shot(asset_index=3)) == {}


def test_an_in_point_becomes_a_seek_before_the_input():
    args = assemble._shot_inputs(shot(source_start=2.5))
    assert args.index("-ss") < args.index("-i")
    assert args[args.index("-ss") + 1] == "2.500"


def test_a_still_is_fed_as_one_frame_when_it_will_be_moved():
    args = assemble._shot_inputs(shot(kind="image", path=Path("/nowhere/x.jpg"),
                                      motion="push", motion_zoom=1.12))
    # -loop 1 here would multiply against zoompan's d= and emit minutes of video
    assert "-loop" not in args


def test_a_still_is_looped_when_it_will_not_move():
    args = assemble._shot_inputs(shot(kind="image", path=Path("/nowhere/x.jpg")))
    assert "-loop" in args and "-t" in args


def test_every_shot_is_cropped_to_the_configured_frame():
    chain = assemble._shot_chain(shot(), "0:v", "v0")
    assert f"crop={config.VIDEO_WIDTH}:{config.VIDEO_HEIGHT}" in chain
    assert "setsar=1" in chain


def test_a_still_reaches_concat_converted_to_limited_range():
    """A full-range JPEG after a limited-range clip crashed ffmpeg 7.1's concat."""
    chain = assemble._shot_chain(
        shot(kind="image", path=Path("/nowhere/x.jpg"), motion="push",
             motion_zoom=1.12), "0:v", "v0")
    steps = chain.split(",")
    # converted before it is labelled - a bare setrange would ship 0-255 as 16-235
    assert steps.index("scale=out_range=tv") < steps.index("setrange=tv")


def test_the_graph_concatenates_every_shot_and_maps_the_audio():
    shots = [shot(index=i, start=i * 3.0) for i in range(4)]
    graph, stage = assemble.build_filter_graph(
        shots, audio_index=4, total=12.5, narration=12.0,
        ass_path=None, fonts_dir=None,
    )
    assert "concat=n=4:v=1:a=0" in graph
    assert "[4:a]" in graph and "[aout]" in graph
    assert "apad" in graph, "the picture outlives the narration"
    assert stage in graph


def test_the_narration_is_loudness_normalised(monkeypatch):
    monkeypatch.setattr(config, "AUDIO_NORMALIZE", True)
    graph, _ = assemble.build_filter_graph(
        [shot()], audio_index=1, total=3.5, narration=3.0,
        ass_path=None, fonts_dir=None,
    )
    chain = [c for c in graph.split(";") if "[aout]" in c][0]

    assert f"loudnorm=I={config.AUDIO_LOUDNESS_TARGET:g}" in chain
    assert f"TP={config.AUDIO_LOUDNESS_PEAK:g}" in chain
    assert f"LRA={config.AUDIO_LOUDNESS_RANGE:g}" in chain


def test_normalisation_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(config, "AUDIO_NORMALIZE", False)
    graph, _ = assemble.build_filter_graph(
        [shot()], audio_index=1, total=3.5, narration=3.0,
        ass_path=None, fonts_dir=None,
    )
    assert "loudnorm" not in graph
    assert f"aresample={config.AUDIO_SAMPLE_RATE}" in graph


def test_the_fade_is_applied_after_normalisation(monkeypatch):
    """loudnorm is dynamic - ahead of it, a fade reads as the audio going quiet
    and is pulled straight back up."""
    monkeypatch.setattr(config, "AUDIO_NORMALIZE", True)
    monkeypatch.setattr(config, "AUDIO_FADE_OUT_SECONDS", 0.35)
    graph, _ = assemble.build_filter_graph(
        [shot()], audio_index=1, total=3.5, narration=3.0,
        ass_path=None, fonts_dir=None,
    )
    chain = [c for c in graph.split(";") if "[aout]" in c][0]

    assert chain.index("loudnorm") < chain.index("afade")
    # loudnorm runs at 192 kHz internally, so come back down before encoding
    assert chain.index("loudnorm") < chain.index("aresample")
    assert chain.index("afade") < chain.index("apad")


def test_captions_are_burned_in_when_an_ass_file_is_given(tmp_path):
    subtitle = tmp_path / "captions.ass"
    subtitle.write_text("", encoding="utf-8")
    graph, _ = assemble.build_filter_graph(
        [shot()], audio_index=1, total=3.5, narration=3.0,
        ass_path=subtitle, fonts_dir=None,
    )
    assert "subtitles=filename=" in graph
    assert "fontsdir" not in graph, "a system font needs no directory scan"


def test_a_project_path_goes_into_the_filter_relative():
    inside = config.BASE_DIR / "output" / "x.ass"
    quoted = assemble._filter_path(inside)

    assert quoted == "'output/x.ass'"
    assert "\\" not in quoted, "a Windows separator would need escaping"


def test_a_path_outside_the_project_has_its_drive_colon_escaped():
    quoted = assemble._filter_path(Path("/somewhere/else/x.ass").resolve())
    if ":" in quoted:
        assert "\\:" in quoted


# ---------------------------------------------------------------------------
# Attribution carried through to steps 6 and 7
# ---------------------------------------------------------------------------

def test_only_footage_that_made_the_cut_is_credited():
    manifest = make_manifest(assets_per_beat=6, beats=3)
    shots, _ = plan(manifest)
    credits = assemble.collect_attribution(manifest, shots)

    used = {shot.asset_index for shot in shots}
    assert {credit["asset_index"] for credit in credits} == used
    assert len(credits) < len(manifest["assets"])


def test_credits_carry_step_4s_fields_and_the_screen_time():
    manifest = make_manifest()
    shots, _ = plan(manifest)
    credit = assemble.collect_attribution(manifest, shots)[0]

    for field in ("provider", "creator", "creator_url", "license",
                  "source_url", "attribution", "attribution_required"):
        assert field in credit
    assert credit["screen_seconds"] > 0


def test_screen_time_adds_up_across_repeats():
    manifest = make_manifest(assets_per_beat=1, beats=1, narration=11.0)
    manifest["assets"][0]["duration"] = 30.0
    shots, _ = plan(manifest, narration=11.0)
    credits = assemble.collect_attribution(manifest, shots)

    assert len(credits) == 1
    assert credits[0]["screen_seconds"] == pytest.approx(
        sum(s.duration for s in shots), abs=0.01
    )


def test_the_attribution_file_flags_what_has_to_be_credited(tmp_path):
    manifest = make_manifest(assets_per_beat=1, beats=1)
    manifest["assets"][0].update(kind="image", attribution_required=True,
                                 duration=0)
    shots, _ = plan(manifest)
    target = tmp_path / "attribution.txt"
    assemble.write_attribution_file(target, manifest,
                                    assemble.collect_attribution(manifest, shots))

    written = target.read_text(encoding="utf-8")
    assert "CREDIT REQUIRED" in written
    assert manifest["topic"] in written


# ---------------------------------------------------------------------------
# Inputs
# ---------------------------------------------------------------------------

def test_a_manifest_can_be_handed_over_as_a_dict_a_file_or_a_directory(tmp_path):
    manifest = make_manifest()
    (tmp_path / "manifest.json").write_text(json.dumps(manifest),
                                            encoding="utf-8")

    assert assemble._resolve_visuals(manifest)["run_id"] == "test-run"
    assert assemble._resolve_visuals(tmp_path)["run_id"] == "test-run"
    assert assemble._resolve_visuals(tmp_path / "manifest.json")["run_id"] \
        == "test-run"


def test_something_that_is_not_a_step_4_manifest_says_so(tmp_path):
    (tmp_path / "manifest.json").write_text('{"hello": 1}', encoding="utf-8")
    with pytest.raises(assemble.AssemblyError, match="step 4 manifest"):
        assemble._resolve_visuals(tmp_path)


def test_a_missing_audio_file_is_named_in_the_error():
    with pytest.raises(assemble.AssemblyError, match="not found"):
        assemble._resolve_audio("nowhere/missing.mp3")


def test_an_asset_resolves_through_its_relative_path_when_the_run_has_moved(tmp_path):
    (tmp_path / "clip.mp4").write_bytes(b"x")
    asset = {"path": "/gone/clip.mp4", "relative_path": "clip.mp4"}

    assert assemble._asset_path(asset, tmp_path) == tmp_path / "clip.mp4"


def test_the_run_directory_is_dated_and_named_for_the_run():
    from datetime import date

    directory = config.video_run_dir("ocean-1a2b", when=date(2026, 8, 19))
    assert directory.parent.name == "2026-08-19"
    assert directory.name == "ocean-1a2b"
    assert directory.parent.parent == config.OUTPUT_DIR


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# Music bed
# ---------------------------------------------------------------------------

@pytest.fixture
def music_dir(monkeypatch, tmp_path):
    """A music library of the suite's own, never the real assets/music/."""
    directory = tmp_path / "music"
    directory.mkdir()
    monkeypatch.setattr(config, "MUSIC_DIR", directory)
    monkeypatch.setattr(config, "MUSIC_USAGE_FILE", directory / "usage.json")
    monkeypatch.setattr(config, "MUSIC_ENABLED", True)
    monkeypatch.setattr(config, "MUSIC_FILE", None)
    return directory


def track_file(directory: Path, name: str, credit: str | None = None) -> Path:
    path = directory / name
    path.write_bytes(b"not really audio")
    if credit is not None:
        path.with_suffix(".txt").write_text(credit, encoding="utf-8")
    return path


def music_graph(total=45.0, narration=44.0):
    graph, _ = assemble.build_filter_graph(
        [shot()], audio_index=1, total=total, narration=narration,
        ass_path=None, fonts_dir=None, music_index=2,
    )
    return graph


# --- the mix ---------------------------------------------------------------

def test_amix_does_not_renormalise(monkeypatch):
    """normalize=1 is the default and would halve the narration."""
    monkeypatch.setattr(config, "MUSIC_DUCKING", False)
    assert "amix=inputs=2:normalize=0" in music_graph()


def test_the_bed_is_loudness_matched_not_left_at_its_own_level(monkeypatch):
    monkeypatch.setattr(config, "MUSIC_DUCKING", False)
    chain = [c for c in music_graph().split(";") if c.endswith("[bed]")][0]
    assert f"loudnorm=I={config.MUSIC_LOUDNESS_TARGET:g}" in chain


def test_the_bed_is_trimmed_to_the_reel(monkeypatch):
    """The input is fed in looping, so something has to make it finite again."""
    monkeypatch.setattr(config, "MUSIC_DUCKING", False)
    chain = [c for c in music_graph(total=12.5).split(";") if c.endswith("[bed]")][0]
    assert "atrim=end=12.500" in chain


def test_the_bed_fades_at_both_ends(monkeypatch):
    monkeypatch.setattr(config, "MUSIC_DUCKING", False)
    monkeypatch.setattr(config, "MUSIC_FADE_IN_SECONDS", 1.0)
    monkeypatch.setattr(config, "MUSIC_FADE_OUT_SECONDS", 1.5)
    chain = [c for c in music_graph(total=20.0).split(";") if c.endswith("[bed]")][0]

    assert "afade=t=in:st=0:d=1" in chain
    assert "afade=t=out:st=18.500:d=1.5" in chain


def test_no_music_index_leaves_the_graph_exactly_as_it_was():
    graph, _ = assemble.build_filter_graph(
        [shot()], audio_index=1, total=10.0, narration=9.0,
        ass_path=None, fonts_dir=None,
    )
    assert "[aout]" in graph
    for absent in ("amix", "sidechaincompress", "[bed]", "asplit"):
        assert absent not in graph


# --- ducking ---------------------------------------------------------------

def test_ducking_splits_the_narration_rather_than_decoding_it_twice(monkeypatch):
    monkeypatch.setattr(config, "MUSIC_DUCKING", True)
    graph = music_graph()

    assert "asplit=2[narr][duckkey]" in graph
    assert graph.count("[1:a]") == 1, "the narration is one input, split in the graph"


def test_the_bed_is_the_thing_compressed_not_the_voice(monkeypatch):
    """sidechaincompress compresses its first input and only listens to the
    second - the wrong way round would duck the narration under the music."""
    monkeypatch.setattr(config, "MUSIC_DUCKING", True)
    chain = [c for c in music_graph().split(";") if "sidechaincompress" in c][0]
    assert chain.startswith("[bed][duckkey]")


def test_ducking_uses_the_configured_compressor(monkeypatch):
    monkeypatch.setattr(config, "MUSIC_DUCKING", True)
    monkeypatch.setattr(config, "MUSIC_DUCK_THRESHOLD", 0.05)
    monkeypatch.setattr(config, "MUSIC_DUCK_RATIO", 6.0)
    chain = [c for c in music_graph().split(";") if "sidechaincompress" in c][0]

    assert "threshold=0.05" in chain
    assert "ratio=6" in chain


def test_ducking_off_mixes_the_bed_flat(monkeypatch):
    monkeypatch.setattr(config, "MUSIC_DUCKING", False)
    graph = music_graph()

    assert "sidechaincompress" not in graph
    assert "asplit" not in graph
    assert "[narr][bed]amix" in graph


def test_a_ratio_of_one_is_not_ducking(monkeypatch):
    """Ratio 1 is a compressor that does nothing - do not pay for the filter."""
    monkeypatch.setattr(config, "MUSIC_DUCKING", True)
    monkeypatch.setattr(config, "MUSIC_DUCK_RATIO", 1.0)
    assert "sidechaincompress" not in music_graph()


# --- choosing a track ------------------------------------------------------

def test_an_empty_library_is_not_an_error(music_dir):
    track, ledger = assemble.resolve_track(None, [])
    assert track is None and ledger is None


def test_only_audio_extensions_are_picked_up(music_dir):
    track_file(music_dir, "good.mp3")
    track_file(music_dir, "notes.md")
    track_file(music_dir, "cover.jpg")
    assert [t.name for t in assemble.available_tracks()] == ["good.mp3"]


def test_a_sidecar_is_read_as_a_required_credit(music_dir):
    track_file(music_dir, "a.mp3", credit='"Night Drive" by A. Composer, CC BY 4.0')
    track_file(music_dir, "b.mp3")
    by_name = {t.name: t for t in assemble.available_tracks()}

    assert by_name["a.mp3"].credit_required is True
    assert by_name["a.mp3"].credit.startswith('"Night Drive"')
    assert by_name["b.mp3"].credit_required is False


def test_an_empty_sidecar_is_not_a_credit(music_dir):
    track_file(music_dir, "a.mp3", credit="   \n  ")
    assert assemble.available_tracks()[0].credit_required is False


def test_music_can_be_switched_off_entirely(music_dir):
    track_file(music_dir, "a.mp3")
    assert assemble.resolve_track(False, []) == (None, None)


def test_disabled_in_config_beats_a_full_library(music_dir, monkeypatch):
    monkeypatch.setattr(config, "MUSIC_ENABLED", False)
    track_file(music_dir, "a.mp3")
    assert assemble.resolve_track(None, []) == (None, None)


def test_a_track_named_by_bare_filename_is_found_in_the_library(music_dir):
    track_file(music_dir, "calm.mp3")
    track, ledger = assemble.resolve_track("calm.mp3", [])

    assert track.name == "calm.mp3"
    assert ledger is None, "a pinned track is not rotating, so nothing to record"


def test_a_track_named_on_the_command_line_must_exist(music_dir):
    """Typed just now - swallowing the typo would hide it."""
    with pytest.raises(assemble.AssemblyError, match="music track not found"):
        assemble.resolve_track("nope.mp3", [])


def test_a_stale_MUSIC_FILE_warns_and_falls_back(music_dir, monkeypatch):
    """Set once, months ago. A moved file should not fail every run from here."""
    monkeypatch.setattr(config, "MUSIC_FILE", "moved-away.mp3")
    track_file(music_dir, "still-here.mp3")
    warnings = []

    track, ledger = assemble.resolve_track(None, warnings)

    assert track.name == "still-here.mp3"
    assert any("moved-away.mp3" in w for w in warnings)


# --- rotation --------------------------------------------------------------

def test_a_library_rotates_instead_of_repeating(music_dir):
    for name in ("a.mp3", "b.mp3", "c.mp3"):
        track_file(music_dir, name)

    picked = []
    for run in range(6):
        track, ledger = assemble.resolve_track(None, [])
        ledger.record(f"run-{run}", track)
        picked.append(track.name)

    assert picked == ["a.mp3", "b.mp3", "c.mp3", "a.mp3", "b.mp3", "c.mp3"]


def test_a_bed_the_cuts_can_land_on_is_picked_over_one_they_cannot(music_dir):
    """The whole point of measuring a tempo is that the rotation uses it."""
    track_file(music_dir, "a-beatless.mp3", "Someone (CC BY 3.0)\nbpm: none")
    track_file(music_dir, "b-gridded.mp3", "Someone (CC BY 3.0)\nbpm: 120")

    track, _ledger = assemble.resolve_track(None, [])

    assert track.name == "b-gridded.mp3"


def test_a_beatless_bed_still_comes_round(music_dir, monkeypatch):
    """A preference, not a filter: a two-bed library must not become one bed.

    The cooldown blocks the gridded bed the run after it plays, which is what
    lets the beatless one through without any of this having to count runs.
    """
    monkeypatch.setattr(config, "MUSIC_REUSE_COOLDOWN_RUNS", 1)
    track_file(music_dir, "a-beatless.mp3", "Someone (CC BY 3.0)\nbpm: none")
    track_file(music_dir, "b-gridded.mp3", "Someone (CC BY 3.0)\nbpm: 120")

    picked = []
    for run in range(4):
        track, ledger = assemble.resolve_track(None, [])
        ledger.record(f"run-{run}", track)
        picked.append(track.name)

    assert picked == ["b-gridded.mp3", "a-beatless.mp3"] * 2


def test_the_grid_preference_is_off_when_nothing_cuts_to_a_grid(music_dir,
                                                               monkeypatch):
    """With snapping off the preference would cost variety and buy nothing."""
    monkeypatch.setattr(config, "SHOT_SNAP_TO_BEAT", False)
    track_file(music_dir, "a-beatless.mp3", "Someone (CC BY 3.0)\nbpm: none")
    track_file(music_dir, "b-gridded.mp3", "Someone (CC BY 3.0)\nbpm: 120")

    track, _ledger = assemble.resolve_track(None, [])

    assert track.name == "a-beatless.mp3"


def test_a_fallback_tempo_gives_every_bed_but_a_beatless_one_a_grid(music_dir,
                                                                   monkeypatch):
    """MUSIC_BPM stands a grid up, so `has_grid` cannot just read `bpm`."""
    monkeypatch.setattr(config, "MUSIC_BPM", 100.0)
    silent = assemble.Track(path=Path("no-sidecar.mp3"))
    measured = assemble.Track(path=Path("measured.mp3"), beatless=True)

    assert silent.has_grid is True
    assert measured.has_grid is False


def test_a_typo_tempo_does_not_attract_the_rotation(music_dir):
    """`prepare_track` drops a 9000 BPM grid, so it must not be preferred."""
    track_file(music_dir, "a-real.mp3", "Someone (CC BY 3.0)\nbpm: 120")
    track_file(music_dir, "b-typo.mp3", "Someone (CC BY 3.0)\nbpm: 9000")

    assert assemble._named_track("b-typo.mp3").has_grid is False

    for run in range(2):
        track, ledger = assemble.resolve_track(None, [])
        ledger.record(f"run-{run}", track)
        if run == 0:
            assert track.name == "a-real.mp3"


def test_one_track_is_still_used_when_the_cooldown_covers_everything(music_dir):
    """A library smaller than the cooldown must not mean no music at all."""
    track_file(music_dir, "only.mp3")

    for run in range(3):
        track, ledger = assemble.resolve_track(None, [])
        assert track is not None and track.name == "only.mp3"
        ledger.record(f"run-{run}", track)


def test_a_rerun_of_one_id_does_not_stack_up_in_the_ledger(music_dir):
    track_file(music_dir, "a.mp3")
    ledger = assemble.MusicLedger()
    track = assemble.available_tracks()[0]

    ledger.record("same-run", track)
    ledger.record("same-run", track)

    assert len(assemble.MusicLedger()._data["history"]) == 1


def test_a_corrupt_ledger_does_not_stop_the_reel(music_dir):
    track_file(music_dir, "a.mp3")
    config.MUSIC_USAGE_FILE.write_text("{ not json", encoding="utf-8")

    track, ledger = assemble.resolve_track(None, [])
    assert track.name == "a.mp3"


# --- credits ---------------------------------------------------------------

def test_the_bed_credit_matches_the_shape_step_6_reads(music_dir):
    track_file(music_dir, "a.mp3", credit="Track by Someone, CC BY 4.0")
    entry = assemble.music_credit_entry(assemble.available_tracks()[0], 45.0)

    # what caption.py's _credit_line and build_credits actually look at
    assert entry["attribution"] == "Track by Someone, CC BY 4.0"
    assert entry["attribution_required"] is True
    assert entry["kind"] == "music"
    assert entry["provider"] == "", "a local file has no provider to credit"


def test_an_uncredited_bed_asks_for_nothing(music_dir):
    track_file(music_dir, "a.mp3")
    entry = assemble.music_credit_entry(assemble.available_tracks()[0], 45.0)
    assert entry["attribution_required"] is False


def test_the_bed_is_not_counted_among_the_footage(tmp_path, music_dir):
    track_file(music_dir, "a.mp3", credit="Track by Someone, CC BY 4.0")
    credits = [
        {"kind": "video", "file": "clip.mp4", "attribution": "a clip",
         "attribution_required": False, "screen_seconds": 3.0},
        assemble.music_credit_entry(assemble.available_tracks()[0], 45.0),
    ]
    path = tmp_path / "attribution.txt"
    assemble.write_attribution_file(path, {"topic": "t", "run_id": "r"}, credits)
    written = path.read_text(encoding="utf-8")

    assert "1 asset(s) appear in this video" in written, "the bed is not on screen"
    assert "Music:" in written
    assert "Track by Someone" in written


# --- probing ---------------------------------------------------------------

def test_an_unreadable_track_is_dropped_rather_than_failing(music_dir):
    """Everything else here degrades; the bed is the least important input."""
    track_file(music_dir, "broken.mp3")       # not real audio
    warnings = []
    assert assemble.prepare_track(assemble.available_tracks()[0], warnings) is None
    assert any("could not be read" in w for w in warnings)


def test_probe_audio_reads_a_file_with_no_video_stream():
    """probe_media calls an audio-only file broken - this is why it is separate."""
    narration = config.AUDIO_DIR / "bioluminescence.mp3"
    if not narration.is_file():
        pytest.skip("no sample narration on this machine")

    assert assemble.probe_audio(narration) > 0
    assert assemble.probe_media(narration).get("ok") is False


def test_probe_audio_on_something_that_is_not_audio():
    assert assemble.probe_audio(config.BASE_DIR / "README.md") == 0.0


# --- the open --------------------------------------------------------------

def test_the_reel_cuts_straight_in_by_default():
    """The first half-second decides whether the viewer stays; a fade spends it
    showing black."""
    assert config.VIDEO_FADE_IN_SECONDS == 0.0

    graph, _ = assemble.build_filter_graph(
        [shot()], audio_index=1, total=10.0, narration=9.0,
        ass_path=None, fonts_dir=None,
    )
    assert "fade=t=in" not in graph
    assert "fade=t=out" in graph, "the tail still fades - by then they have stayed"


def test_a_soft_open_is_still_available(monkeypatch):
    monkeypatch.setattr(config, "VIDEO_FADE_IN_SECONDS", 0.4)
    graph, _ = assemble.build_filter_graph(
        [shot()], audio_index=1, total=10.0, narration=9.0,
        ass_path=None, fonts_dir=None,
    )
    assert "fade=t=in:st=0:d=0.4" in graph


# ---------------------------------------------------------------------------
# Pacing: the reel does not cut at one rate throughout
# ---------------------------------------------------------------------------

def test_the_curve_opens_fast_settles_and_tightens_again():
    total = 45.0
    opening = assemble.shot_target_at(0.0, total)
    settled = assemble.shot_target_at(total / 2, total)
    closing = assemble.shot_target_at(total - 0.5, total)

    assert opening == pytest.approx(config.SHOT_HOOK_SECONDS, abs=0.01)
    assert settled == pytest.approx(config.SHOT_TARGET_SECONDS, abs=0.01)
    assert opening < settled
    assert closing < settled


def test_the_curve_rises_monotonically_out_of_the_opening():
    total = 60.0
    ramp = config.SHOT_PACING_RAMP_SECONDS
    samples = [assemble.shot_target_at(ramp * i / 10, total) for i in range(11)]
    for earlier, later in zip(samples, samples[1:]):
        assert later >= earlier - 1e-9


def test_a_flat_setting_ignores_the_position_entirely(monkeypatch):
    monkeypatch.setattr(config, "SHOT_PACING", "flat")
    for position in (0.0, 5.0, 20.0, 44.0):
        assert assemble.shot_target_at(position, 45.0) == config.SHOT_TARGET_SECONDS


def test_an_opening_beat_gets_more_cuts_than_the_same_beat_would_mid_reel():
    """The count comes from integrating the curve across the window, so a long
    opening beat is not handed the steady rate just because its midpoint is
    past the ramp."""
    opening = assemble._shot_count(15.0, 8, position=0.0, total=45.0)
    middle = assemble._shot_count(15.0, 8, position=15.0, total=45.0)
    assert opening > middle


def test_shot_spans_lengthen_through_the_opening_and_fill_the_window_exactly():
    spans = assemble._shot_spans(0.0, 15.0, 6, 45.0)
    lengths = [round(end - start, 3) for start, end in spans]

    assert spans[0][0] == 0.0
    assert spans[-1][1] == 15.0
    for earlier, later in zip(lengths, lengths[1:]):
        assert later >= earlier - 0.01, lengths
    assert sum(lengths) == pytest.approx(15.0, abs=0.01)
    for earlier, later in zip(spans, spans[1:]):
        assert earlier[1] == pytest.approx(later[0], abs=0.001)


def test_one_shot_spans_the_whole_window():
    assert assemble._shot_spans(2.0, 7.0, 1, 45.0) == [(2.0, 7.0)]


# ---------------------------------------------------------------------------
# Snapping: cuts land on words, not on the arithmetic
# ---------------------------------------------------------------------------

def test_a_cut_mid_word_is_moved_into_the_neighbouring_silence():
    # A cut wants to fall at 3.0, which is halfway through "darkness".
    words = words_from([
        ("total", 2.0, 2.6), ("darkness", 2.8, 3.4), ("sunlight", 3.9, 4.5),
    ])
    plan_rows = [
        {"start": 0.0, "end": 3.0},
        {"start": 3.0, "end": 6.0},
    ]
    moved = assemble._snap_boundaries(plan_rows, words)

    assert moved == 1
    # Two silences: 2.6-2.8 and 3.4-3.9. Only the first is inside
    # SHOT_SNAP_MAX_SECONDS of the cut, so the cut goes to its midpoint.
    assert plan_rows[0]["end"] == pytest.approx(2.7, abs=0.01)
    assert plan_rows[1]["start"] == plan_rows[0]["end"]


def test_a_cut_already_sitting_in_a_gap_is_left_alone():
    words = words_from([("one", 1.0, 1.4), ("two", 2.0, 2.4)])
    plan_rows = [{"start": 0.0, "end": 1.7}, {"start": 1.7, "end": 4.0}]

    assert assemble._snap_boundaries(plan_rows, words) == 0
    assert plan_rows[0]["end"] == 1.7


def test_a_cut_is_never_moved_further_than_the_configured_limit():
    # The only gap is at 8.0, far outside SHOT_SNAP_MAX_SECONDS of the cut.
    words = words_from([("one", 7.0, 7.5), ("two", 8.5, 9.0)])
    plan_rows = [{"start": 0.0, "end": 3.0}, {"start": 3.0, "end": 12.0}]

    assert assemble._snap_boundaries(plan_rows, words) == 0
    assert plan_rows[0]["end"] == 3.0


def test_snapping_never_collapses_a_shot():
    """A nudge that would leave either neighbour under the floor is declined,
    however good the landing would have been."""
    # The gap at 3.2-3.3 is well within reach of the cut at 3.0, but taking it
    # would leave the second shot 0.35s long.
    words = words_from([("a", 2.6, 3.2), ("b", 3.3, 3.6)])
    plan_rows = [{"start": 0.0, "end": 3.0}, {"start": 3.0, "end": 3.6}]

    assert assemble._snap_boundaries(plan_rows, words) == 0
    assert plan_rows[0]["end"] == 3.0


def test_the_open_and_the_close_are_never_moved():
    words = words_from([("a", 0.3, 0.5), ("b", 1.4, 1.6), ("c", 2.6, 2.9)])
    plan_rows = [
        {"start": 0.0, "end": 1.0},
        {"start": 1.0, "end": 2.0},
        {"start": 2.0, "end": 4.0},
    ]
    assemble._snap_boundaries(plan_rows, words)

    assert plan_rows[0]["start"] == 0.0
    assert plan_rows[-1]["end"] == 4.0


def test_a_moved_cut_becomes_the_floor_for_the_next_one():
    words = words_from([("a", 0.9, 1.1), ("b", 1.9, 2.1), ("c", 3.0, 3.2)])
    plan_rows = [
        {"start": 0.0, "end": 1.0},
        {"start": 1.0, "end": 2.0},
        {"start": 2.0, "end": 5.0},
    ]
    assemble._snap_boundaries(plan_rows, words)

    for earlier, later in zip(plan_rows, plan_rows[1:]):
        assert earlier["end"] == later["start"]
        assert later["start"] > earlier["start"]


def test_a_very_short_silence_is_not_treated_as_somewhere_to_cut():
    """What looks like a gap can be the stop consonant inside one phrase."""
    gaps, starts = assemble._word_cut_points(
        words_from([("hot", 1.0, 1.5), ("pot", 1.52, 2.0)])
    )
    assert gaps == []
    assert starts == [1.0, 1.52]


def test_words_with_no_timings_at_all_are_survived():
    plan_rows = [{"start": 0.0, "end": 3.0}, {"start": 3.0, "end": 6.0}]
    assert assemble._snap_boundaries(plan_rows, [{"text": "x"}]) == 0
    assert assemble._snap_boundaries(plan_rows, []) == 0


def test_the_timeline_reports_how_many_cuts_it_moved():
    manifest = make_manifest(assets_per_beat=2, beats=3)
    words = words_from([(f"w{i}", i * 0.5, i * 0.5 + 0.3) for i in range(90)])
    shots, warnings = plan(manifest, words=words)

    assert any("nudged onto a word boundary" in warning for warning in warnings)
    assert shots[0].start == 0.0
    for earlier, later in zip(shots, shots[1:]):
        assert earlier.end == pytest.approx(later.start, abs=0.01)


def test_snapping_off_leaves_the_cuts_where_the_arithmetic_put_them(monkeypatch):
    manifest = make_manifest(assets_per_beat=2, beats=3)
    words = words_from([(f"w{i}", i * 0.5, i * 0.5 + 0.3) for i in range(90)])

    monkeypatch.setattr(config, "SHOT_SNAP_TO_WORDS", False)
    without = [s.start for s in plan(manifest, words=words)[0]]
    monkeypatch.setattr(config, "SHOT_SNAP_TO_WORDS", True)
    with_snap = [s.start for s in plan(manifest, words=words)[0]]

    assert without != with_snap
    assert plan(manifest)[0][0].duration == pytest.approx(without[1], abs=0.01)


def test_a_snapped_shot_still_decides_looping_from_its_final_length():
    """Looping depends on the slot the clip has to fill, and snapping changes
    that - so the decision has to come after the nudge, not before it."""
    manifest = make_manifest(assets_per_beat=1, beats=2)
    for asset in manifest["assets"]:
        asset["duration"] = 7.0
    words = words_from([(f"w{i}", i * 0.4, i * 0.4 + 0.25) for i in range(110)])
    shots, _ = plan(manifest, words=words)

    for shot_ in shots:
        if not shot_.looped:
            assert shot_.source_start + shot_.duration <= 7.0 + 0.05


# ---------------------------------------------------------------------------
# The beat grid: the opening cuts belong to the music
# ---------------------------------------------------------------------------

def test_a_bpm_line_is_read_as_a_setting_and_never_as_a_credit(music_dir):
    """The load-bearing half. The credit is copied verbatim into
    attribution.txt and from there into the public post text by step 6, so a
    setting left in it is published as part of somebody's name."""
    track_file(music_dir, "a.mp3",
               credit='"Autumn prelude" by zero-project (CC BY 3.0)\nbpm: 92\n')
    track = assemble.available_tracks()[0]

    assert track.credit == '"Autumn prelude" by zero-project (CC BY 3.0)'
    assert "bpm" not in track.credit.lower()
    assert track.bpm == 92.0
    assert track.credit_required is True


def test_a_sidecar_with_no_settings_reads_exactly_as_it_did(music_dir):
    track_file(music_dir, "a.mp3", credit='"Night Drive" by A. Composer, CC BY 4.0')
    track = assemble.available_tracks()[0]

    assert track.credit == '"Night Drive" by A. Composer, CC BY 4.0'
    assert track.bpm == 0.0 and track.beat_offset == 0.0


def test_the_settings_syntax_is_forgiving_about_spacing(music_dir):
    track_file(music_dir, "a.mp3", credit="Someone\nBPM=128\n  offset : 0.25  \n")
    track = assemble.available_tracks()[0]

    assert (track.bpm, track.beat_offset) == (128.0, 0.25)
    assert track.credit == "Someone"


def test_a_track_can_say_it_has_no_pulse_at_all(music_dir):
    """"bpm: none" is measured; no bpm line at all is nobody having looked.

    Two of the five beds in assets/music/ have no constant grid in them - a
    line fit to their tracked beats leaves 39-2018ms of residual against
    7-14ms for the three that do, and 110ms for pure noise. No number can fix
    that, so the sidecar says so and the run stops asking.
    """
    track_file(music_dir, "a.mp3",
               credit='"Plathe (Beatless Ambient Mix)" by POA (CC BY 3.0)\n'
                      "bpm: none\n")
    track = assemble.available_tracks()[0]

    assert track.beatless is True
    assert track.bpm == 0.0
    assert track.credit == '"Plathe (Beatless Ambient Mix)" by POA (CC BY 3.0)'
    assert "bpm" not in track.credit.lower()


def test_a_beatless_track_is_not_asked_for_a_tempo_every_run(music_dir,
                                                             monkeypatch):
    monkeypatch.setattr(config, "SHOT_SNAP_TO_BEAT", True)
    monkeypatch.setattr(config, "MUSIC_BPM", 0.0)
    monkeypatch.setattr(assemble, "probe_audio", lambda path: 90.0)
    track_file(music_dir, "a.mp3", credit="Someone\nbpm: none\n")

    warnings: list[str] = []
    track = assemble.prepare_track(assemble.available_tracks()[0], warnings)

    assert track.bpm == 0.0
    assert any("beatless" in warning for warning in warnings)
    assert not any("add a" in warning for warning in warnings)


def test_a_track_nobody_has_measured_is_still_asked_for_a_tempo(music_dir,
                                                                monkeypatch):
    monkeypatch.setattr(config, "SHOT_SNAP_TO_BEAT", True)
    monkeypatch.setattr(config, "MUSIC_BPM", 0.0)
    monkeypatch.setattr(assemble, "probe_audio", lambda path: 90.0)
    track_file(music_dir, "a.mp3", credit="Someone")

    warnings: list[str] = []
    assemble.prepare_track(assemble.available_tracks()[0], warnings)

    assert any("no tempo is known" in warning for warning in warnings)


def test_a_beatless_track_snaps_nothing(music_dir):
    track = assemble.Track(path=Path("a.mp3"), beatless=True)

    assert assemble.beat_grid(track, 6.0) == []


def test_the_grid_runs_from_the_reels_first_frame():
    """The bed is fed in with -stream_loop and trimmed from 0, so the track's
    first sample is the reel's first frame and beat one is t=0."""
    track = assemble.Track(path=Path("a.mp3"), bpm=120.0)
    assert assemble.beat_grid(track, 2.0) == [0.0, 0.5, 1.0, 1.5, 2.0]


def test_a_first_downbeat_inside_the_file_shifts_the_whole_grid():
    track = assemble.Track(path=Path("a.mp3"), bpm=120.0, beat_offset=0.2)
    assert assemble.beat_grid(track, 1.5) == [0.2, 0.7, 1.2]


def test_any_downbeat_will_do_not_only_the_first():
    """0.2 and 1.7 are the same downbeat three beats apart at 120 BPM."""
    early = assemble.Track(path=Path("a.mp3"), bpm=120.0, beat_offset=0.2)
    late = assemble.Track(path=Path("a.mp3"), bpm=120.0, beat_offset=1.7)
    assert assemble.beat_grid(early, 3.0) == assemble.beat_grid(late, 3.0)


def test_a_track_with_no_tempo_written_down_has_no_grid():
    assert assemble.beat_grid(assemble.Track(path=Path("a.mp3")), 30.0) == []
    assert assemble.beat_grid(None, 30.0) == []


def test_music_bpm_stands_in_for_a_track_with_no_sidecar(music_dir, monkeypatch):
    monkeypatch.setattr(assemble, "probe_audio", lambda path: 60.0)
    monkeypatch.setattr(config, "MUSIC_BPM", 120.0)
    track_file(music_dir, "a.mp3")
    track = assemble.prepare_track(assemble.available_tracks()[0], [])

    assert track.bpm == 120.0, "the track carries the tempo that was used"
    assert assemble.beat_grid(track, 1.0) == [0.0, 0.5, 1.0]


def test_a_track_states_its_own_tempo_over_the_fallback(music_dir, monkeypatch):
    monkeypatch.setattr(assemble, "probe_audio", lambda path: 60.0)
    monkeypatch.setattr(config, "MUSIC_BPM", 120.0)
    track_file(music_dir, "a.mp3", credit="Someone\nbpm: 60\n")
    track = assemble.prepare_track(assemble.available_tracks()[0], [])

    assert track.bpm == 60.0
    assert assemble.beat_grid(track, 1.0) == [0.0, 1.0]


def test_an_implausible_tempo_is_refused_with_a_warning(music_dir, monkeypatch):
    """A tempo is typed by hand into a text file, so it is checked like one."""
    monkeypatch.setattr(assemble, "probe_audio", lambda path: 60.0)
    track_file(music_dir, "a.mp3", credit="Someone\nbpm: 9000\n")
    warnings = []
    track = assemble.prepare_track(assemble.available_tracks()[0], warnings)

    assert track is not None, "a bad tempo must not cost the reel its music"
    assert track.bpm == 0.0
    assert any("9000 BPM" in warning for warning in warnings)
    assert assemble.beat_grid(track, 10.0) == []


def test_a_feature_that_cannot_run_says_so(music_dir, monkeypatch):
    """SHOT_SNAP_TO_BEAT on and no tempo anywhere is a silent no-op.

    run.py prints step warnings and not assemble's own summary, so on the
    daily path this looked switched on and cut nothing on the grid. Every reel
    shipped so far reports 0 of N cuts on beat for exactly this reason.
    """
    monkeypatch.setattr(assemble, "probe_audio", lambda path: 60.0)
    monkeypatch.setattr(config, "MUSIC_BPM", 0.0)
    monkeypatch.setattr(config, "SHOT_SNAP_TO_BEAT", True)
    track_file(music_dir, "millennium-sunset.mp3", credit="Someone")
    warnings = []
    track = assemble.prepare_track(assemble.available_tracks()[0], warnings)

    assert track is not None, "a missing tempo must not cost the reel its music"
    assert any("no tempo is known" in w for w in warnings)
    assert any("millennium-sunset.txt" in w for w in warnings)


def test_no_tempo_is_not_worth_mentioning_when_snapping_is_off(
    music_dir, monkeypatch
):
    monkeypatch.setattr(assemble, "probe_audio", lambda path: 60.0)
    monkeypatch.setattr(config, "MUSIC_BPM", 0.0)
    monkeypatch.setattr(config, "SHOT_SNAP_TO_BEAT", False)
    track_file(music_dir, "a.mp3", credit="Someone")
    warnings = []
    assemble.prepare_track(assemble.available_tracks()[0], warnings)

    assert not any("no tempo" in w for w in warnings)


def test_a_track_with_a_tempo_does_not_warn(music_dir, monkeypatch):
    monkeypatch.setattr(assemble, "probe_audio", lambda path: 60.0)
    monkeypatch.setattr(config, "SHOT_SNAP_TO_BEAT", True)
    track_file(music_dir, "a.mp3", credit="Someone\nbpm: 92\n")
    warnings = []
    assemble.prepare_track(assemble.available_tracks()[0], warnings)

    assert not any("no tempo" in w for w in warnings)


def test_an_opening_cut_is_placed_on_the_beat():
    plan_rows = [{"start": 0.0, "end": 3.2}, {"start": 3.2, "end": 7.0}]
    beats = assemble.beat_grid(assemble.Track(path=Path("a.mp3"), bpm=120.0), 8.0)

    assert assemble._snap_to_beats(plan_rows, beats, []) == [1]
    assert plan_rows[0]["end"] == pytest.approx(3.0, abs=0.001)
    assert plan_rows[1]["start"] == plan_rows[0]["end"]


def test_a_beat_in_the_middle_of_a_word_is_refused(monkeypatch):
    """Cutting mid-word is the defect snapping exists to prevent, and the music
    does not get to override it - the boundary falls through to the words."""
    monkeypatch.setattr(config, "SHOT_BEAT_MAX_SECONDS", 0.25)
    words = words_from([("darkness", 2.9, 3.1)])
    plan_rows = [{"start": 0.0, "end": 3.2}, {"start": 3.2, "end": 7.0}]
    beats = assemble.beat_grid(assemble.Track(path=Path("a.mp3"), bpm=120.0), 8.0)

    assert assemble._snap_to_beats(plan_rows, beats, words) == []
    assert plan_rows[0]["end"] == 3.2, "left exactly where it was"


def test_a_beat_on_the_attack_of_a_word_is_a_fine_landing(monkeypatch):
    """A cut on the first frame of a word is the fallback `_snap_boundaries`
    already reaches for by choice; only the middle of one is out."""
    monkeypatch.setattr(config, "SHOT_BEAT_MAX_SECONDS", 0.25)
    words = words_from([("darkness", 3.0, 3.6)])
    plan_rows = [{"start": 0.0, "end": 3.2}, {"start": 3.2, "end": 7.0}]
    beats = assemble.beat_grid(assemble.Track(path=Path("a.mp3"), bpm=120.0), 8.0)

    assert assemble._snap_to_beats(plan_rows, beats, words) == [1]
    assert plan_rows[0]["end"] == pytest.approx(3.0, abs=0.001)


def test_cuts_past_the_window_are_left_to_the_words(monkeypatch):
    monkeypatch.setattr(config, "SHOT_BEAT_WINDOW_SECONDS", 8.0)
    plan_rows = [{"start": 0.0, "end": 12.2}, {"start": 12.2, "end": 16.0}]
    beats = assemble.beat_grid(assemble.Track(path=Path("a.mp3"), bpm=120.0), 20.0)

    assert assemble._snap_to_beats(plan_rows, beats, []) == []
    assert plan_rows[0]["end"] == 12.2


def test_the_word_pass_leaves_a_beat_snapped_cut_alone():
    """Moving it onto a nearer pause is the same as taking it off the beat."""
    words = words_from([("total", 2.0, 2.6), ("darkness", 2.8, 3.4)])
    plan_rows = [{"start": 0.0, "end": 3.0}, {"start": 3.0, "end": 6.0}]

    assert assemble._snap_boundaries(plan_rows, words, skip=[1]) == 0
    assert plan_rows[0]["end"] == 3.0


def test_no_grid_plans_the_timeline_exactly_as_it_always_did():
    manifest = make_manifest()
    words = words_from([(f"w{i}", i * 0.4, i * 0.4 + 0.25) for i in range(110)])

    without = [(s.start, s.duration) for s in plan(manifest, words=words)[0]]
    empty = [(s.start, s.duration)
             for s in plan(manifest, words=words, beats=[])[0]]
    assert without == empty


def test_the_beat_pass_runs_before_the_words_and_wins():
    """Both passes want the same boundaries. Run the other way round, the word
    pass finds a nearer pause for every cut the music just placed and the grid
    ends up having done nothing at all."""
    manifest = make_manifest()
    words = words_from([(f"w{i}", i * 0.4, i * 0.4 + 0.25) for i in range(110)])
    beats = assemble.beat_grid(assemble.Track(path=Path("a.mp3"), bpm=120.0),
                               NARRATION)

    shots, notes = plan(manifest, words=words, beats=beats)
    opening = [s.start for s in shots[1:]
               if s.start <= config.SHOT_BEAT_WINDOW_SECONDS]

    assert opening, "the fixture has to cut inside the window for this to mean anything"
    assert any(assemble._on_grid(beats, start) for start in opening)
    assert any("beat grid" in note for note in notes)


def test_the_grid_is_ignored_when_the_setting_is_off(monkeypatch):
    manifest = make_manifest()
    words = words_from([(f"w{i}", i * 0.4, i * 0.4 + 0.25) for i in range(110)])
    beats = assemble.beat_grid(assemble.Track(path=Path("a.mp3"), bpm=120.0),
                               NARRATION)
    without = [(s.start, s.duration) for s in plan(manifest, words=words)[0]]

    monkeypatch.setattr(config, "SHOT_SNAP_TO_BEAT", False)
    off = [(s.start, s.duration)
           for s in plan(manifest, words=words, beats=beats)[0]]
    assert off == without


# ---------------------------------------------------------------------------
# The hook card
# ---------------------------------------------------------------------------

def ass_with_hook(text: str, seconds: float = 2.2, **overrides) -> str:
    cues = [Cue(words_from([("salt", 0.2, 0.6)]), 0.2, 0.6)]
    return assemble.build_ass(cues, "Arial Black", None,
                              hook=text, hook_seconds=seconds)


def test_the_hook_card_is_a_top_anchored_style_of_its_own():
    body = ass_with_hook("The ocean never gets saltier")
    style = next(line for line in body.splitlines()
                 if line.startswith("Style: Hook,"))
    fields = style.split(",")

    assert fields[-5] == "8", "alignment 8 is top-centre"
    assert fields[-2] == str(config.HOOK_CARD_MARGIN_V)
    assert fields[2] == str(config.HOOK_CARD_FONT_SIZE)


def test_the_hook_card_holds_from_the_first_frame():
    body = ass_with_hook("Nobody has seen it", seconds=2.0)
    event = next(line for line in body.splitlines() if ",Hook," in line)

    assert event.startswith("Dialogue: 0,0:00:00.00,0:00:02.00,Hook,")


def test_the_hook_card_fades_at_both_ends():
    body = ass_with_hook("Nobody has seen it")
    event = next(line for line in body.splitlines() if ",Hook," in line)
    assert f"\\fad({config.HOOK_CARD_FADE_MS},{config.HOOK_CARD_FADE_MS})" in event


def test_no_hook_means_the_ass_is_exactly_what_it_always_was():
    cues = [Cue(words_from([("salt", 0.2, 0.6)]), 0.2, 0.6)]
    plain = assemble.build_ass(cues, "Arial Black", None)

    assert "Style: Hook," not in plain
    assert ",Hook," not in plain
    assert plain == assemble.build_ass(cues, "Arial Black", None,
                                       hook="", hook_seconds=2.2)
    assert plain == assemble.build_ass(cues, "Arial Black", None,
                                       hook="something", hook_seconds=0.0)


def test_the_hook_card_is_uppercased_when_configured(monkeypatch):
    monkeypatch.setattr(config, "HOOK_CARD_UPPERCASE", True)
    assert "NOBODY HAS SEEN IT" in ass_with_hook("Nobody has seen it")
    monkeypatch.setattr(config, "HOOK_CARD_UPPERCASE", False)
    assert "Nobody has seen it" in ass_with_hook("Nobody has seen it")


def test_a_hook_card_cannot_escape_into_ass_markup():
    body = ass_with_hook("{\\an5}drop\\Ntable")
    event = next(line for line in body.splitlines() if ",Hook," in line)
    text = event.split(",,", 1)[1]

    assert "{\\an5}" not in text
    assert "\\N" not in text


def test_a_hook_card_too_wide_for_two_lines_is_scaled_down():
    long_card = "Extraordinarily unpronounceable bioluminescence phenomenon"
    small = assemble._hook_card_font_size(long_card, None)
    assert small < config.HOOK_CARD_FONT_SIZE

    assert assemble._hook_card_font_size("Short", None) == config.HOOK_CARD_FONT_SIZE


def test_the_hook_card_gets_two_lines_before_it_shrinks():
    """A headline of a few words wraps; shrinking it to stay on one row would
    make the biggest text in the reel the smallest thing about it."""
    text = "The ocean never gets any saltier"
    one_line = assemble._fit_font_size(
        text, None, base=config.HOOK_CARD_FONT_SIZE,
        ink=2 * config.HOOK_CARD_OUTLINE_WIDTH,
        margin_h=config.CAPTION_MARGIN_H, lines=1,
    )
    two_lines = assemble._hook_card_font_size(text, None)
    assert two_lines > one_line


def test_a_card_is_derived_from_the_script_when_the_manifest_has_none():
    card = assemble._derive_hook_card(
        "Imagine living in total darkness. Sunlight dies at 650 feet."
    )
    assert card
    assert "sunlight" not in card.lower()
    assert len(card.split()) <= config.HOOK_CARD_MAX_WORDS


# ---------------------------------------------------------------------------
# Sound design
# ---------------------------------------------------------------------------

@pytest.fixture
def sfx_dir(monkeypatch, tmp_path):
    """An effects library of the suite's own, never the real assets/sfx/."""
    directory = tmp_path / "sfx"
    directory.mkdir()
    monkeypatch.setattr(config, "SFX_DIR", directory)
    monkeypatch.setattr(config, "SFX_ENABLED", True)
    monkeypatch.setattr(assemble, "probe_peak_db", lambda path: -6.0)
    return directory


def sfx_file(directory: Path, name: str) -> Path:
    path = directory / name
    path.write_bytes(b"RIFFfake")
    return path


def test_the_role_is_read_off_the_front_of_the_filename():
    for name, role in (("whoosh.wav", "whoosh"), ("whoosh-02.wav", "whoosh"),
                       ("whoosh_soft.mp3", "whoosh"), ("impact.wav", "impact"),
                       ("03-whoosh.wav", "")):
        assert assemble._sfx_role(Path(name)) == role


def test_the_library_groups_files_by_role(sfx_dir):
    sfx_file(sfx_dir, "whoosh.wav")
    sfx_file(sfx_dir, "whoosh-02.wav")
    sfx_file(sfx_dir, "impact.wav")
    sfx_file(sfx_dir, "notes.txt")

    library = assemble.available_sfx()
    assert [p.name for p in library["whoosh"]] == ["whoosh-02.wav", "whoosh.wav"]
    assert [p.name for p in library["impact"]] == ["impact.wav"]
    assert "notes" not in library


def test_each_one_shot_is_levelled_to_the_configured_peak(sfx_dir, monkeypatch):
    sfx_file(sfx_dir, "whoosh.wav")
    monkeypatch.setattr(assemble, "probe_peak_db", lambda path: -3.0)
    warnings: list[str] = []

    effect = assemble.resolve_sfx(None, warnings)["whoosh"][0]
    assert effect.gain_db == pytest.approx(config.SFX_PEAK_TARGET + 3.0, abs=0.01)
    assert not warnings


def test_a_hot_and_a_quiet_one_shot_end_up_in_the_same_place(sfx_dir, monkeypatch):
    """The whole reason the peak is measured rather than assumed."""
    sfx_file(sfx_dir, "whoosh.wav")
    sfx_file(sfx_dir, "whoosh-02.wav")
    peaks = {"whoosh.wav": -1.0, "whoosh-02.wav": -18.0}
    monkeypatch.setattr(assemble, "probe_peak_db",
                        lambda path: peaks[Path(path).name])

    effects = assemble.resolve_sfx(None, [])["whoosh"]
    landed = [effect.peak_db + effect.gain_db for effect in effects]
    assert all(value == pytest.approx(config.SFX_PEAK_TARGET, abs=0.01)
               for value in landed)


def test_a_silent_file_is_dropped_with_a_warning_not_amplified(sfx_dir, monkeypatch):
    sfx_file(sfx_dir, "whoosh.wav")
    monkeypatch.setattr(assemble, "probe_peak_db", lambda path: -91.0)
    warnings: list[str] = []

    assert assemble.resolve_sfx(None, warnings) == {}
    assert any("no signal in it" in warning for warning in warnings)


def test_an_unmeasurable_file_is_dropped_with_a_warning(sfx_dir, monkeypatch):
    sfx_file(sfx_dir, "impact.wav")
    monkeypatch.setattr(assemble, "probe_peak_db", lambda path: None)
    warnings: list[str] = []

    assert assemble.resolve_sfx(None, warnings) == {}
    assert any("could not be measured" in warning for warning in warnings)


def test_a_misnamed_role_is_reported_rather_than_silently_ignored(sfx_dir):
    sfx_file(sfx_dir, "wooosh.wav")
    warnings: list[str] = []

    assert assemble.resolve_sfx(None, warnings) == {}
    assert any("no role this step places" in warning for warning in warnings)


def test_an_empty_library_is_not_a_problem_worth_a_warning(sfx_dir):
    warnings: list[str] = []
    assert assemble.resolve_sfx(None, warnings) == {}
    assert warnings == []


def test_effects_can_be_switched_off_entirely(sfx_dir, monkeypatch):
    sfx_file(sfx_dir, "whoosh.wav")
    assert assemble.resolve_sfx(False, []) == {}
    monkeypatch.setattr(config, "SFX_ENABLED", False)
    assert assemble.resolve_sfx(None, []) == {}


def one_shot(role: str, name: str = "x.wav") -> assemble.Sfx:
    return assemble.Sfx(role=role, path=Path(name), peak_db=-6.0, gain_db=-8.0)


def test_the_impact_lands_on_the_first_frame():
    shots = [shot(index=i, start=i * 4.0, duration=4.0) for i in range(4)]
    cues = assemble.plan_sfx(shots, 16.0, {"impact": [one_shot("impact")]})

    assert len(cues) == 1
    assert cues[0].at == 0.0
    assert cues[0].sfx.role == "impact"


def test_a_whoosh_rides_every_cut_but_never_the_opening_frame():
    shots = [shot(index=i, start=i * 4.0, duration=4.0) for i in range(4)]
    cues = assemble.plan_sfx(shots, 16.0, {"whoosh": [one_shot("whoosh")]})

    assert len(cues) == 3, "three cuts between four shots"
    assert all(cue.at > 0 for cue in cues)


def test_a_whoosh_is_placed_early_so_its_transient_lands_on_the_cut():
    shots = [shot(index=0, start=0.0, duration=4.0),
             shot(index=1, start=4.0, duration=4.0)]
    cues = assemble.plan_sfx(shots, 8.0, {"whoosh": [one_shot("whoosh")]})

    assert cues[0].at == pytest.approx(4.0 - config.SFX_LEAD_SECONDS, abs=0.001)


def test_cuts_crowded_together_get_one_effect_between_them(monkeypatch):
    monkeypatch.setattr(config, "SFX_MIN_GAP_SECONDS", 2.0)
    shots = [shot(index=i, start=i * 0.8, duration=0.8) for i in range(10)]
    cues = assemble.plan_sfx(shots, 8.0, {"whoosh": [one_shot("whoosh")]})

    times = [cue.at for cue in cues]
    for earlier, later in zip(times, times[1:]):
        assert later - earlier >= 2.0 - 0.01


def test_the_impact_holds_the_open_against_an_immediate_cut(monkeypatch):
    """The effect at 0 counts towards the spacing, so a cut 0.4s in does not
    fire a whoosh on top of it."""
    monkeypatch.setattr(config, "SFX_MIN_GAP_SECONDS", 1.6)
    shots = [shot(index=0, start=0.0, duration=0.4),
             shot(index=1, start=0.4, duration=4.0)]
    cues = assemble.plan_sfx(shots, 5.0, {"impact": [one_shot("impact")],
                                          "whoosh": [one_shot("whoosh")]})

    assert [cue.sfx.role for cue in cues] == ["impact"]


def test_several_whooshes_are_rotated_rather_than_one_being_reused():
    shots = [shot(index=i, start=i * 4.0, duration=4.0) for i in range(5)]
    library = {"whoosh": [one_shot("whoosh", "a.wav"), one_shot("whoosh", "b.wav")]}
    cues = assemble.plan_sfx(shots, 20.0, library)

    names = [cue.sfx.name for cue in cues]
    assert len(set(names)) == 2, names


def test_effects_are_delayed_into_place_and_mixed_over_the_finished_audio():
    cues = [assemble.SfxCue(sfx=one_shot("impact"), at=0.0, reason="the open"),
            assemble.SfxCue(sfx=one_shot("whoosh"), at=3.92, reason="cut")]
    graph, _ = assemble.build_filter_graph(
        [shot()], audio_index=1, total=10.0, narration=9.0,
        ass_path=None, fonts_dir=None, sfx_cues=cues, sfx_index=2,
    )

    assert "adelay=3920:all=1" in graph
    assert "[abed][sfx0][sfx1]amix=inputs=3:normalize=0:duration=first[aout]" in graph
    assert "volume=-8.00dB" in graph


def test_an_effect_at_zero_needs_no_delay():
    cues = [assemble.SfxCue(sfx=one_shot("impact"), at=0.0, reason="the open")]
    graph, _ = assemble.build_filter_graph(
        [shot()], audio_index=1, total=10.0, narration=9.0,
        ass_path=None, fonts_dir=None, sfx_cues=cues, sfx_index=2,
    )
    assert "[2:a]" in graph and "adelay" not in graph


def test_effects_ride_over_the_ducking_rather_than_through_it(monkeypatch):
    """A compressor keyed off the narration would pull each transient down at
    exactly the moment it fired."""
    monkeypatch.setattr(config, "MUSIC_DUCKING", True)
    cues = [assemble.SfxCue(sfx=one_shot("whoosh"), at=2.0, reason="cut")]
    graph, _ = assemble.build_filter_graph(
        [shot()], audio_index=1, total=10.0, narration=9.0,
        ass_path=None, fonts_dir=None, music_index=2,
        sfx_cues=cues, sfx_index=3,
    )

    assert "sidechaincompress" in graph
    assert "[ducked]amix=inputs=2:normalize=0:duration=first[abed]" in graph
    assert graph.endswith("[aout]")
    # The effect is not one of the sidechain's inputs.
    assert "[sfx0][duckkey]" not in graph


def test_no_effects_leaves_the_audio_graph_untouched():
    plain, _ = assemble.build_filter_graph(
        [shot()], audio_index=1, total=10.0, narration=9.0,
        ass_path=None, fonts_dir=None,
    )
    with_none, _ = assemble.build_filter_graph(
        [shot()], audio_index=1, total=10.0, narration=9.0,
        ass_path=None, fonts_dir=None, sfx_cues=[], sfx_index=2,
    )
    assert plain == with_none
    assert "abed" not in plain
    assert "sfx" not in plain


# ---------------------------------------------------------------------------
# The cover frame
# ---------------------------------------------------------------------------

def cover_manifest(**overrides) -> dict:
    """A manifest whose assets carry the descriptive fields step 4 now keeps."""
    manifest = make_manifest(assets_per_beat=2, beats=3)
    for index, asset in enumerate(manifest["assets"]):
        asset["source_url"] = (
            f"https://www.pexels.com/video/blue-ocean-clip-{index}-99{index}/"
        )
        asset["query"] = "deep ocean"
    manifest.update(overrides)
    return manifest


def plan_a_cover(shots=None, manifest=None, probes=None, *, card="Hook card",
                 card_seconds=2.2, source=None, looks=None, warnings=None):
    manifest = manifest if manifest is not None else cover_manifest()
    if shots is None:
        shots, _ = plan(manifest)
    probes = probes if probes is not None else probes_for(manifest)
    return assemble.plan_cover(
        shots, manifest, probes, card=card, card_seconds=card_seconds,
        font_file=None, source=source, looks=looks,
        warnings=warnings if warnings is not None else [],
    )


def verdict(grade=2, ok=True, why="a clear shot"):
    return {"grade": grade, "ok": ok, "why": why}


def test_an_aspect_is_read_as_width_and_height():
    assert assemble.parse_aspect("4:5") == (4, 5)
    assert assemble.parse_aspect("1:1") == (1, 1)
    assert assemble.parse_aspect("9x16") == (9, 16)


def test_an_unreadable_aspect_falls_back_to_the_whole_frame():
    top, bottom = assemble.safe_band("not an aspect")
    assert (top, bottom) == (0.0, float(assemble.CAPTION_REFERENCE_HEIGHT))


def test_the_safe_band_is_the_centre_crop_of_the_reference_frame():
    # 1080x1920 cropped to 4:5 is 1080x1350, centred: 285px of frame each end.
    assert assemble.safe_band("4:5") == (285.0, 1635.0)
    # ...and to 1:1 is 1080x1080, which takes 420px off the top.
    assert assemble.safe_band("1:1") == (420.0, 1500.0)


def test_the_default_hook_card_clears_a_four_by_five_crop():
    clears, why = assemble.card_clears_crop("Snailfish: extreme pressure",
                                            None, "4:5")
    assert clears and why == ""


def test_a_square_crop_would_cut_the_top_off_the_hook_card(monkeypatch):
    monkeypatch.setattr(config, "HOOK_CARD_MARGIN_V", 320)
    clears, why = assemble.card_clears_crop("Snailfish: extreme pressure",
                                            None, "1:1")
    assert not clears
    assert "320px" in why and "420px" in why


def test_a_card_low_enough_survives_a_square_crop(monkeypatch):
    monkeypatch.setattr(config, "HOOK_CARD_MARGIN_V", 460)
    clears, _ = assemble.card_clears_crop("Short", None, "1:1")
    assert clears


# --- what the picture shows -------------------------------------------------

def test_a_provider_slug_is_read_as_the_words_in_it():
    assert assemble._describe("medusa-blue-ocean-13693981") == "medusa blue ocean"


def test_the_asset_id_is_not_mistaken_for_part_of_the_description():
    assert not any(char.isdigit()
                   for char in assemble._describe("lions-walking-safari-37551601"))


def test_words_about_the_file_rather_than_the_picture_are_dropped():
    assert assemble._describe("vertical video of a jellyfish") == "a jellyfish"
    assert assemble._describe("jellyfish footage") == "jellyfish"


def test_a_comma_separated_tag_list_survives_readably():
    assert assemble._describe("ocean,  waves,sea") == "ocean, waves, sea"


def test_an_apostrophe_the_slug_dropped_is_put_back():
    """The cover of the mole-rat reel of 2026-09-12, as Pexels spelled it."""
    assert assemble._describe(
        "a-doctor-using-tablet-while-looking-the-patient-s-results"
    ) == "a doctor using tablet while looking the patient's results"
    assert assemble._describe("it-doesn-t-work") == "it doesn't work"


def test_a_bare_t_that_heads_a_word_is_left_alone():
    """"t-shirt" and "t-cell" are stock-library staples, and neither is a
    contraction. Attaching every orphan "t" backwards spelled them "a't
    shirt" and "and't cells" - in alt text, which is read aloud."""
    assert assemble._describe("a-t-shirt") == "a t shirt"
    assert assemble._describe("vitamin-c-and-t-cells") == "vitamin c and t cells"
    # Both rules on one slug: the possessive is restored, the compound is not
    # touched.
    assert assemble._describe("the-dog-s-t-shirt") == "the dog's t shirt"


def test_the_providers_own_title_is_preferred_over_everything_else():
    described, source = assemble._cover_depiction({
        "title": "A jellyfish drifting in open water",
        "source_url": "https://www.pexels.com/video/medusa-blue-ocean-13693981/",
        "query": "snailfish deep ocean",
    })
    assert described == "A jellyfish drifting in open water"
    assert source == "provider_title"


def test_the_source_url_describes_the_clip_when_there_is_no_title():
    described, source = assemble._cover_depiction({
        "title": "",
        "source_url": "https://www.pexels.com/video/medusa-blue-ocean-13693981/",
        "query": "snailfish deep ocean",
    })
    assert described == "medusa blue ocean"
    assert source == "source_url"


def test_the_query_is_the_last_resort_because_it_is_what_was_asked_for():
    described, source = assemble._cover_depiction({"query": "snailfish deep ocean"})
    assert described == "snailfish deep ocean"
    assert source == "query"


def test_the_beats_image_prompt_is_never_used_to_describe_the_cover():
    """Step 4's prompt says what was wanted. Stock search does not always agree.

    On a real run "a snailfish swims in the deep ocean" was filled by a clip of
    a jellyfish - so describing the cover from it would put a confident, wrong
    sentence in front of someone who cannot check it.
    """
    described, source = assemble._cover_depiction({
        "image_prompt": "A snailfish swims in the deep ocean.",
        "source_url": "https://www.pexels.com/video/medusa-blue-ocean-1/",
    })
    assert "snailfish" not in described
    assert source == "source_url"


# --- which shot -------------------------------------------------------------

def test_a_shot_that_held_the_screen_outranks_a_glimpse():
    probe = {"width": 1080, "height": 1920, "ok": True}
    long_shot = assemble.cover_score(shot(duration=3.2), probe)
    brief = assemble.cover_score(shot(duration=0.6), probe)
    assert long_shot > brief


def test_a_looped_or_reused_shot_is_marked_down():
    probe = {"width": 1080, "height": 1920, "ok": True}
    plain = assemble.cover_score(shot(duration=3.0), probe)
    assert assemble.cover_score(shot(duration=3.0, looped=True), probe) < plain
    assert assemble.cover_score(shot(duration=3.0, reused=True), probe) < plain


def test_footage_that_needs_no_crop_is_worth_more_than_footage_that_does():
    portrait = assemble.cover_score(
        shot(duration=3.0), {"width": 1080, "height": 1920, "ok": True})
    landscape = assemble.cover_score(
        shot(duration=3.0), {"width": 3840, "height": 2160, "ok": True})
    assert portrait > landscape


def test_the_top_of_the_reel_is_preferred_to_the_end_of_it():
    probe = {"width": 1080, "height": 1920, "ok": True}
    early = assemble.cover_score(shot(duration=3.0, start=1.0), probe)
    late = assemble.cover_score(shot(duration=3.0, start=40.0), probe)
    assert early > late


# --- and what the frame actually shows --------------------------------------

def test_a_frame_nobody_looked_at_scores_what_it_always_did():
    probe = {"width": 1080, "height": 1920, "ok": True}
    assert (assemble.cover_score(shot(), probe, None)
            == assemble.cover_score(shot(), probe))
    assert (assemble.cover_score(shot(), probe, {})
            == assemble.cover_score(shot(), probe))


def test_a_better_looking_frame_outranks_a_duller_one():
    probe = {"width": 1080, "height": 1920, "ok": True}
    striking = assemble.cover_score(shot(), probe, verdict(grade=3))
    flat = assemble.cover_score(shot(), probe, verdict(grade=1))
    assert striking > flat


def test_the_look_decides_between_candidates_the_proxies_cannot_separate():
    # The case that actually comes up: the candidates are all early shots of
    # similar length and resolution, so every blind term is close to a tie and
    # the only thing left to choose on is what the frames show.
    probe = {"width": 1080, "height": 1920, "ok": True}
    dull = shot(index=0, duration=3.0, start=0.0)
    striking = shot(index=1, duration=2.8, start=3.0)

    assert assemble.cover_score(dull, probe) > assemble.cover_score(striking, probe)
    assert (assemble.cover_score(striking, probe, verdict(grade=3))
            > assemble.cover_score(dull, probe, verdict(grade=1)))


def test_the_look_is_a_term_and_not_a_gate():
    # Deliberately bounded. The model sees one frame; the proxies know things
    # about the clip that one frame does not show, so a looped 0.8s scrap it
    # likes must not beat a 4s native-vertical hero shot it is lukewarm about.
    # If this ever needs to flip, COVER_LOOK_WEIGHT is the knob - not the rule.
    ideal = shot(duration=4.0, start=0.0)
    scrap = shot(index=9, duration=0.8, start=12.0, looped=True, reused=True)
    good = {"width": 1080, "height": 1920, "ok": True}
    poor = {"width": 3840, "height": 2160, "ok": True}

    assert assemble.cover_score(ideal, good) > assemble.cover_score(scrap, poor)
    assert (assemble.cover_score(ideal, good, verdict(grade=1))
            > assemble.cover_score(scrap, poor, verdict(grade=3)))


def test_a_vetoed_frame_falls_behind_one_that_was_merely_dull():
    probe = {"width": 1080, "height": 1920, "ok": True}
    vetoed = assemble.cover_score(shot(), probe, verdict(grade=2, ok=False))
    dull = assemble.cover_score(shot(), probe, verdict(grade=0, ok=True))
    assert vetoed < dull


def footage_pool(shots):
    """The candidates for a cover taken from the footage, not the hook card.

    The card pool on this fixture is one shot deep - the card runs 2.2s and the
    opening shot is longer than that - so the footage pool is the one with
    anything to choose between.
    """
    return assemble.cover_candidates(
        shots, card="", card_seconds=0.0, font_file=None,
    )["pool"]


def test_the_cover_the_model_likes_is_the_one_that_ships():
    shots, _ = plan(cover_manifest())
    pool = footage_pool(shots)
    assert len(pool) > 1

    # Hand the last candidate the only good grade and it should win, even
    # though `cover_score` prefers the top of the reel on every other term.
    last = pool[-1]
    blind = plan_a_cover(shots=shots, card="", card_seconds=0.0)
    assert blind["shot_index"] != last.index

    looks = {assemble.cover_label(one): verdict(grade=0 if one is not last else 3)
             for one in pool}
    seen = plan_a_cover(shots=shots, card="", card_seconds=0.0, looks=looks)
    assert seen["shot_index"] == last.index


def test_a_cover_nobody_could_ship_is_still_shipped_and_said_out_loud():
    shots, _ = plan(cover_manifest())
    looks = {assemble.cover_label(one): verdict(grade=0, ok=False, why="too dark")
             for one in footage_pool(shots)}

    warnings = []
    cover = plan_a_cover(shots=shots, card="", card_seconds=0.0, looks=looks,
                         warnings=warnings)
    assert cover is not None
    assert any("would not ship any candidate" in note for note in warnings)
    assert any("too dark" in note for note in warnings)


def card_shortlist(shots, **overrides):
    kwargs = dict(card="Hook card", card_seconds=2.2, font_file=None)
    kwargs.update(overrides)
    return assemble.cover_candidates(shots, **kwargs)


def test_the_card_pool_is_shallower_than_the_look_is_allowed_to_see():
    # The reason the fallback exists. COVER_LOOK_MAX_CANDIDATES is 6, but the
    # card runs 2.2s, so the pool under it is one or two shots deep and the
    # cap has nothing to cap. A reel that opens on the wrong image has no
    # second candidate, and the choice falls through to proxies that know the
    # clip's size and slot and nothing about what is in it.
    shots, _ = plan(cover_manifest())
    under = card_shortlist(shots)["pool"]
    # The pool the fallback builds, which keeps the real card_seconds - not
    # `footage_pool`, which asks as though there were no card at all and so
    # still offers the shots underneath one.
    footage = card_shortlist(shots, source="footage")["pool"]

    assert len(under) < config.COVER_LOOK_MAX_CANDIDATES
    assert len(footage) > len(under), "the footage pool is the deeper one"
    assert {one.index for one in under} & {one.index for one in footage} == set(), \
        "the two pools are disjoint - the footage pool starts after the card"


def test_a_card_pool_the_model_rejects_outright_is_exhausted():
    shots, _ = plan(cover_manifest())
    shortlist = card_shortlist(shots)
    looks = {assemble.cover_label(one): verdict(grade=1, ok=False,
                                                why="misleading")
             for one in shortlist["pool"]}

    assert assemble.cover_look_exhausted(shortlist, looks)


def test_one_frame_worth_shipping_is_enough_to_keep_the_card():
    shots, _ = plan(cover_manifest())
    shortlist = card_shortlist(shots)
    pool = shortlist["pool"]
    looks = {assemble.cover_label(one): verdict(grade=1, ok=False)
             for one in pool}
    looks[assemble.cover_label(pool[0])] = verdict(grade=2, ok=True)

    assert not assemble.cover_look_exhausted(shortlist, looks)


def test_an_ungraded_pool_is_not_a_rejected_one():
    # No verdicts means the look did not happen - a failed call, or frames
    # that would not render. Re-grading a pool nobody graded would spend a
    # second model call to learn the same nothing.
    shots, _ = plan(cover_manifest())
    assert not assemble.cover_look_exhausted(card_shortlist(shots), {})


def test_asking_for_the_card_outright_survives_the_model_disliking_it():
    # COVER_SOURCE=hook_card is a caller that asked for the card knowing what
    # it costs. A rejection is not a licence to overrule it.
    shots, _ = plan(cover_manifest())
    shortlist = card_shortlist(shots, source="hook_card")
    looks = {assemble.cover_label(one): verdict(grade=0, ok=False)
             for one in shortlist["pool"]}

    assert shortlist["want_card"]
    assert not assemble.cover_look_exhausted(shortlist, looks)


def test_a_footage_cover_is_never_asked_to_fall_back_to_itself():
    shots, _ = plan(cover_manifest())
    shortlist = assemble.cover_candidates(shots, card="", card_seconds=0.0,
                                          font_file=None)
    looks = {assemble.cover_label(one): verdict(grade=0, ok=False)
             for one in shortlist["pool"]}

    assert not shortlist["want_card"]
    assert not assemble.cover_look_exhausted(shortlist, looks)


def test_the_footage_cover_drops_the_card_rather_than_pasting_it_later():
    # What the fallback hands `plan_cover`. The card cannot ride along: a
    # headline over footage the viewer never saw it over is a frame the reel
    # does not contain, which is the one thing the cover is not allowed to be.
    shots, _ = plan(cover_manifest())
    cover = plan_a_cover(shots=shots, source="footage")

    assert cover["source"] == "footage"
    assert cover["hook_card_text"] == ""
    assert cover["shot_index"] in {one.index for one in footage_pool(shots)}


def stub_the_cover_look(monkeypatch, verdicts_by_round):
    """Stand in for the encoder and the model. Returns the rounds it was asked.

    `verdicts_by_round` is one {label: verdict} per call to `judge_covers`, so
    a test says what the model thinks of the card pool and then of the footage
    pool without rendering a frame or spending a call.
    """
    rounds: list[list[str]] = []

    def fake_preview(pool, **kwargs):
        return [(assemble.cover_label(one), b"") for one in pool], []

    def fake_judge(frames, **kwargs):
        labels = [label for label, _ in frames]
        rounds.append(labels)
        answers = verdicts_by_round[len(rounds) - 1]
        return {label: answers[label] for label in labels if label in answers}, []

    monkeypatch.setattr(assemble, "preview_cover_frames", fake_preview)
    monkeypatch.setattr(assemble.look, "judge_covers", fake_judge)
    return rounds


def look_at(shots, **overrides):
    kwargs = dict(card="Hook card", card_seconds=2.2, font_name="Arial",
                  font_file=None, fonts_dir=None, topic="deep ocean",
                  source=None)
    kwargs.update(overrides)
    return assemble.look_at_covers(shots, **kwargs)


def test_a_refused_card_pool_sends_the_look_to_the_footage(monkeypatch):
    shots, _ = plan(cover_manifest())
    card = card_shortlist(shots)["pool"]
    footage = card_shortlist(shots, source="footage")["pool"]
    rounds = stub_the_cover_look(monkeypatch, [
        {assemble.cover_label(one): verdict(grade=1, ok=False) for one in card},
        {assemble.cover_label(one): verdict(grade=3, ok=True) for one in footage},
    ])

    source, verdicts, warnings = look_at(shots)

    assert len(rounds) == 2, "the footage pool was graded too"
    assert rounds[0] == [assemble.cover_label(one) for one in card]
    assert rounds[1] == [assemble.cover_label(one) for one in footage]
    assert source == "footage"
    assert set(verdicts) == {assemble.cover_label(one) for one in footage}
    assert any("taken from the footage instead" in note for note in warnings)


def test_a_card_pool_the_model_will_ship_is_never_looked_past(monkeypatch):
    shots, _ = plan(cover_manifest())
    card = card_shortlist(shots)["pool"]
    rounds = stub_the_cover_look(monkeypatch, [
        {assemble.cover_label(one): verdict(grade=3, ok=True) for one in card},
    ])

    source, verdicts, warnings = look_at(shots)

    assert len(rounds) == 1, "the second pass costs a call and was not needed"
    assert source is None, "plan_cover is left to resolve COVER_SOURCE itself"
    assert set(verdicts) == {assemble.cover_label(one) for one in card}


def test_footage_the_model_also_refuses_leaves_the_card_where_it_was(monkeypatch):
    shots, _ = plan(cover_manifest())
    card = card_shortlist(shots)["pool"]
    footage = card_shortlist(shots, source="footage")["pool"]
    refused = [
        {assemble.cover_label(one): verdict(grade=1, ok=False) for one in card},
        {assemble.cover_label(one): verdict(grade=0, ok=False) for one in footage},
    ]
    rounds = stub_the_cover_look(monkeypatch, refused)

    source, verdicts, warnings = look_at(shots)

    assert len(rounds) == 2
    assert source is None, "kept the card rather than swapping on a tie"
    assert set(verdicts) == {assemble.cover_label(one) for one in card}
    assert any("as well as the hook card" in note for note in warnings)


def test_a_look_that_never_answered_is_not_reported_as_a_refusal(monkeypatch):
    # An empty second round is a failed call or a frame that would not render,
    # not the model turning the footage down. Saying otherwise would put a
    # verdict nobody gave into the run log.
    shots, _ = plan(cover_manifest())
    card = card_shortlist(shots)["pool"]
    stub_the_cover_look(monkeypatch, [
        {assemble.cover_label(one): verdict(grade=1, ok=False) for one in card},
        {},
    ])

    source, _, warnings = look_at(shots)

    assert source is None
    assert any("no second opinion" in note for note in warnings)
    assert not any("rejected the footage" in note for note in warnings)


def test_the_cover_records_what_it_beat_and_not_only_what_it_is():
    shots, _ = plan(cover_manifest())
    pool = footage_pool(shots)
    looks = {assemble.cover_label(one): verdict(grade=index % 4)
             for index, one in enumerate(pool)}

    cover = plan_a_cover(shots=shots, card="", card_seconds=0.0, looks=looks)
    assert cover["look"] is not None
    assert len(cover["look_candidates"]) == len(pool)
    assert {entry["shot_index"] for entry in cover["look_candidates"]} == {
        one.index for one in pool
    }
    # Every candidate's grade is on the record, not only the winner's - which
    # is what makes it possible to ask later whether the look changed anything.
    assert all("grade" in entry for entry in cover["look_candidates"])


def test_a_run_with_no_look_records_no_candidates_rather_than_empty_ones():
    cover = plan_a_cover()
    assert cover["look"] is None
    assert cover["look_candidates"] == []


def test_the_shortlist_the_look_sees_is_the_shortlist_that_is_ranked():
    # Two callers compute this pool - the runner, to render previews, and
    # `plan_cover`, to rank them. A verdict keyed off a different shortlist
    # would be a verdict about a frame that was never in the running.
    shots, _ = plan(cover_manifest())
    kwargs = dict(card="Hook card", card_seconds=2.2, font_file=None)
    first = assemble.cover_candidates(shots, **kwargs)
    second = assemble.cover_candidates(shots, **kwargs)
    assert [one.index for one in first["pool"]] == [one.index for one in second["pool"]]
    assert first["want_card"] == second["want_card"]

    chosen = plan_a_cover(shots=shots)
    assert chosen["shot_index"] in {one.index for one in first["pool"]}


def test_the_frame_a_candidate_is_previewed_at_is_the_frame_that_ships():
    # `preview_cover_frames` and `plan_cover` both go through
    # `cover_frame_time`, so the graded picture is the one that gets written.
    one = shot(duration=4.0, start=0.0, source_start=1.5)
    assert (assemble.cover_frame_time(one, card_seconds=2.2, want_card=False)
            == assemble.cover_frame_time(one, card_seconds=2.2, want_card=False))

    cover = plan_a_cover(shots=[one], card="", card_seconds=0.0)
    _, source_time = assemble.cover_frame_time(one, card_seconds=0.0,
                                               want_card=False)
    assert cover["source_time"] == source_time


def test_a_still_is_previewed_from_its_only_frame():
    one = shot(kind="image", duration=3.0, source_start=0.0)
    _, source_time = assemble.cover_frame_time(one, card_seconds=0.0,
                                               want_card=False)
    assert source_time == 0.0


def test_frames_are_sent_smaller_than_they_ship_but_the_same_shape():
    width, height = assemble._look_size()
    assert width == config.LOOK_FRAME_WIDTH
    assert height % 2 == 0
    assert abs(width / height - config.VIDEO_WIDTH / config.VIDEO_HEIGHT) < 0.02


def test_the_beats_reach_step_5b_as_the_lines_their_footage_plays_under():
    """What step 5b needs to judge a frame against the words over it.

    Beats rather than caption cues: a cue is three words long, so a frame would
    land against a fragment with no claim in it, while a beat is the sentence
    whose claim the footage was chosen to carry.
    """
    manifest = {"beats": [
        {"index": 0, "start": 0.0, "end": 18.652,
         "text": "Two billion years ago, shallow water washed over sand."},
        {"index": 1, "start": 18.652, "end": 26.515,
         "text": "This was a planet of bare continents, before the first\n"
                 "   animal had even evolved."},
    ]}
    lines = assemble.narration_lines(manifest)

    assert lines[0][:2] == (0.0, 18.652)
    # Wrapped text arrives as one line, or the grader reads a line break as a
    # sentence ending and judges the frame against half a claim.
    assert lines[1][2] == ("This was a planet of bare continents, before the "
                           "first animal had even evolved.")


def test_a_beat_missing_its_timings_is_dropped_not_guessed_at():
    """A wrong time sends the grader to a line that was not playing."""
    manifest = {"beats": [
        {"start": 0.0, "end": 4.0, "text": "timed"},
        {"start": None, "end": 9.0, "text": "no start"},
        {"end": 12.0, "text": "no start key at all"},
        {"start": 12.0, "end": 15.0, "text": "   "},
        "not a dict",
    ]}
    assert assemble.narration_lines(manifest) == [(0.0, 4.0, "timed")]


def test_a_manifest_with_no_beats_asks_for_no_timings():
    """Step 5b then grades from the script alone, as it did before."""
    assert assemble.narration_lines({}) == []


def test_nothing_is_sampled_out_of_a_reel_that_is_not_there(tmp_path):
    frames, notes = assemble.sample_reel_frames(tmp_path / "gone.mp4", 40.0)
    assert frames == [] and notes == []


# --- which shot a verdict is about ------------------------------------------
#
# The timeline in assembly.json is the plan. The render drifts from it: each
# shot is encoded as a whole number of frames and the rounding accumulates, so
# by the middle of a reel a cut can be several frames from where the plan puts
# it. A grade is only aimable if the frame's label and the timeline agree
# about which clip was on screen, and near a cut they did not.

def test_a_sample_landing_on_a_cut_is_pulled_back_inside_its_shot():
    """The giant squid reel, 2026-09-19, exactly.

    The sample fell at 20.53s. The plan says 18.90-20.61s is "serene ocean
    waves in sunlight"; the render had already cut, and the model came back
    "a small shark swimming over a reef", graded 0 of 3. Anything acting on
    that verdict would have replaced the ocean and left the shark up.
    """
    waves = shot(index=8, beat_index=4, start=18.90, duration=1.71)
    assert assemble._shot_at([waves], 20.5325) is waves
    at = assemble._inside_shot(waves, 20.5325)
    assert at == pytest.approx(20.36, abs=0.001)
    assert at <= 20.61 - assemble._LOOK_SHOT_EDGE_SECONDS


def test_a_sample_well_inside_its_shot_is_left_where_it_fell():
    """Even coverage is the point of the sampling; the pull is only a nudge."""
    long_shot = shot(index=3, beat_index=1, start=10.0, duration=6.0)
    assert assemble._inside_shot(long_shot, 13.0) == 13.0


def test_a_shot_too_short_to_have_an_interior_is_sampled_at_its_middle():
    """Both margins overlap, so the midpoint is the furthest from either cut."""
    brief = shot(index=0, beat_index=0, start=4.0, duration=0.3)
    assert assemble._inside_shot(brief, 4.29) == pytest.approx(4.15)


def test_a_sample_past_the_last_shot_belongs_to_no_shot():
    """The tail is nobody's footage, and must not be blamed on the last clip."""
    only = shot(index=0, beat_index=0, start=0.0, duration=3.0)
    assert assemble._shot_at([only], 3.4) is None
    assert assemble._shot_at(None, 1.0) is None


def _stub_frame_grabs(monkeypatch):
    """An ffmpeg that writes a byte to the path it was asked for.

    The sampling under test is arithmetic - which timestamp, belonging to
    which shot - so the encoder is the one part of it worth faking.
    """
    monkeypatch.setattr(assemble, "ffmpeg_binary", lambda: "ffmpeg")

    def fake_run(args, **kwargs):
        Path(args[-1]).write_bytes(b"jpeg")
        return assemble.subprocess.CompletedProcess(args, 0, "", "")

    monkeypatch.setattr(assemble.subprocess, "run", fake_run)


def test_each_sampled_frame_records_the_shot_it_shows(tmp_path, monkeypatch):
    """Written down here, so no later reader re-derives it from a timestamp."""
    video = tmp_path / "reel.mp4"
    video.write_bytes(b"not really an mp4")
    shots = [shot(index=0, beat_index=0, start=0.0, duration=5.0),
             shot(index=1, beat_index=3, start=5.0, duration=5.0)]

    _stub_frame_grabs(monkeypatch)
    placement: dict[str, dict] = {}
    frames, notes = assemble.sample_reel_frames(
        video, 10.0, count=2, shots=shots, placement=placement,
    )

    assert notes == []
    assert [label for label, _ in frames] == ["2.5s", "7.5s"]
    assert placement == {"2.5s": {"shot": 0, "beat": 0},
                         "7.5s": {"shot": 1, "beat": 3}}


def test_the_grade_carries_the_shot_each_verdict_is_about():
    """What `placement` is for: the verdict and the clip, in one place."""
    record = {"graded": True, "grade": 1, "problems": ["off_topic"],
              "summary": "a shark under a line about beaks",
              "frames": [{"frame": "20.4s", "grade": 0, "shows": "a shark"},
                         {"frame": "26.4s", "grade": 1, "shows": "a stone"}]}
    assemble._place_verdicts(record, {"20.4s": {"shot": 8, "beat": 4},
                                      "26.4s": {"shot": 11, "beat": 5}})

    assert record["frames"][0] == {"frame": "20.4s", "grade": 0,
                                   "shows": "a shark", "shot": 8, "beat": 4}
    assert record["frames"][1]["shot"] == 11


def test_a_verdict_with_no_placement_is_left_alone():
    """No shot list to ask means no answer, which is not the same as shot 0."""
    record = {"frames": [{"frame": "2.5s", "grade": 2, "shows": "a reef"}]}
    assemble._place_verdicts(record, {})
    assert record["frames"] == [{"frame": "2.5s", "grade": 2, "shows": "a reef"}]
    # And a record nothing graded survives being offered placements.
    blank = assemble.look.blank_grade()
    assemble._place_verdicts(blank, {"2.5s": {"shot": 0, "beat": 0}})
    assert blank == assemble.look.blank_grade()


def test_sampling_still_works_with_no_shot_list_to_ask(tmp_path, monkeypatch):
    """`shots` is an improvement to a label, not a thing a caller must have."""
    video = tmp_path / "reel.mp4"
    video.write_bytes(b"not really an mp4")

    _stub_frame_grabs(monkeypatch)
    frames, notes = assemble.sample_reel_frames(video, 10.0, count=2)
    assert [label for label, _ in frames] == ["2.5s", "7.5s"] and notes == []


# --- the choice -------------------------------------------------------------

def test_auto_opens_on_the_hook_card_when_the_crop_allows_it():
    cover = plan_a_cover(source="auto")
    assert cover["source"] == "hook_card"
    assert cover["hook_card_text"] == "Hook card"
    assert "clears a centre crop" in cover["reason"]
    assert cover["card_in_safe_band"] is True


def test_auto_falls_back_to_footage_when_the_crop_would_cut_the_card(monkeypatch):
    monkeypatch.setattr(config, "COVER_SAFE_ASPECT", "1:1")
    warnings: list[str] = []
    cover = plan_a_cover(source="auto", warnings=warnings)
    assert cover["source"] == "footage"
    assert cover["hook_card_text"] == ""
    assert cover["card_in_safe_band"] is False
    assert any("rather than the hook card" in w for w in warnings)


def test_hook_card_keeps_the_card_even_where_the_crop_bites(monkeypatch):
    monkeypatch.setattr(config, "COVER_SAFE_ASPECT", "1:1")
    cover = plan_a_cover(source="hook_card")
    assert cover["source"] == "hook_card"
    assert "kept it anyway" in cover["reason"]


def test_footage_never_takes_the_frame_from_under_the_card():
    cover = plan_a_cover(source="footage", card_seconds=2.2)
    assert cover["source"] == "footage"
    assert cover["hook_card_text"] == ""
    assert cover["time"] >= 2.2


def test_a_reel_with_no_card_takes_its_cover_from_the_footage():
    cover = plan_a_cover(card="", card_seconds=0.0, source="auto")
    assert cover["source"] == "footage"
    assert cover["card_in_safe_band"] is None


def test_asking_for_a_card_that_does_not_exist_says_so():
    warnings: list[str] = []
    cover = plan_a_cover(card="", card_seconds=0.0, source="hook_card",
                         warnings=warnings)
    assert cover["source"] == "footage"
    assert any("no hook card" in w for w in warnings)


def test_an_unknown_cover_source_is_reported_and_treated_as_auto():
    warnings: list[str] = []
    cover = plan_a_cover(source="whatever", warnings=warnings)
    assert cover["mode"] == "auto"
    assert any("unknown COVER_SOURCE" in w for w in warnings)


def test_covers_can_be_switched_off_entirely(monkeypatch):
    monkeypatch.setattr(config, "COVER_ENABLED", False)
    assert plan_a_cover() is None


def test_the_card_frame_is_taken_while_the_card_is_still_up():
    cover = plan_a_cover(source="hook_card", card_seconds=2.2)
    assert 0 < cover["time"] < 2.2


def test_the_frame_is_taken_away_from_both_cuts_of_its_shot():
    manifest = cover_manifest()
    shots, _ = plan(manifest)
    cover = plan_a_cover(shots=shots, manifest=manifest, source="footage")
    chosen = next(s for s in shots if s.index == cover["shot_index"])
    assert cover["time"] > chosen.start
    assert cover["time"] < chosen.end


def test_a_clip_is_seeked_to_the_same_in_point_the_edit_uses():
    manifest = cover_manifest()
    shots, _ = plan(manifest)
    cover = plan_a_cover(shots=shots, manifest=manifest, source="footage")
    chosen = next(s for s in shots if s.index == cover["shot_index"])
    assert cover["source_time"] == pytest.approx(
        chosen.source_start + (cover["time"] - chosen.start), abs=0.002
    )


def test_a_still_has_only_one_frame_to_take():
    manifest = cover_manifest()
    for asset in manifest["assets"]:
        asset["kind"] = "image"
        asset["duration"] = 0.0
    shots, _ = plan(manifest)
    cover = plan_a_cover(shots=shots, manifest=manifest, source="footage")
    assert cover["kind"] == "image"
    assert cover["source_time"] == 0.0


def test_only_shots_near_the_top_of_the_reel_are_candidates(monkeypatch):
    monkeypatch.setattr(config, "COVER_SEARCH_SECONDS", 6.0)
    manifest = cover_manifest()
    shots, _ = plan(manifest)
    cover = plan_a_cover(shots=shots, manifest=manifest, source="footage")
    assert cover["time"] < 6.0


def test_the_cover_is_planned_but_not_written_until_something_writes_it():
    cover = plan_a_cover()
    assert cover["written"] is False
    assert cover["path"] is None
    assert cover["bytes"] == 0


def test_the_record_says_which_shot_and_why():
    cover = plan_a_cover()
    for key in ("source", "mode", "reason", "time", "shot_index", "beat_index",
                "asset_index", "file", "kind", "depicts", "depicts_source",
                "safe_aspect", "score"):
        assert key in cover, key
    assert cover["reason"]


def test_a_cover_frame_is_the_size_of_the_reels_own_frame():
    cover = plan_a_cover()
    assert cover["width"] == config.VIDEO_WIDTH
    assert cover["height"] == config.VIDEO_HEIGHT


def test_footage_nothing_describes_is_reported_rather_than_left_blank():
    manifest = cover_manifest()
    for asset in manifest["assets"]:
        asset["title"] = ""
        asset["source_url"] = ""
        asset["query"] = ""
    warnings: list[str] = []
    cover = plan_a_cover(manifest=manifest, warnings=warnings)
    # The topic is the last thing standing, and the run says so.
    assert cover["depicts_source"] == "topic"
    assert any("nothing describes" in w for w in warnings)


# --- the card the cover is drawn through ------------------------------------

def test_the_covers_card_is_drawn_without_the_reels_fade():
    """One frame at t=0 is the middle of the reel's fade-in."""
    faded = assemble.build_ass([], "Arial Black", None,
                               hook="Hook card", hook_seconds=1.0)
    flat = assemble.build_ass([], "Arial Black", None, hook="Hook card",
                              hook_seconds=1.0, hook_fade_ms=0)
    assert "\\fad(" in faded
    assert "\\fad(" not in flat
    assert "HOOK CARD" in flat


def test_the_covers_card_is_built_by_the_same_code_as_the_reels():
    """So the headline on the cover cannot drift from the one in the video."""
    def card_style(ass: str) -> list[str]:
        return [line for line in ass.splitlines()
                if line.startswith("Style: Hook")]

    reel = assemble.build_ass([], "Arial Black", None,
                              hook="A longer hook card that wraps",
                              hook_seconds=2.2, hook_fade_ms=0)
    cover = assemble.build_ass([], "Arial Black", None,
                               hook="A longer hook card that wraps",
                               hook_seconds=1.0, hook_fade_ms=0)
    assert card_style(reel) == card_style(cover)


# ---------------------------------------------------------------------------
# Motion - how the frame moves across a shot
# ---------------------------------------------------------------------------

@pytest.fixture
def moving(monkeypatch):
    """The shipped defaults, pinned so a config change cannot rewrite a test."""
    monkeypatch.setattr(config, "SHOT_MOTION", "alternate")
    monkeypatch.setattr(config, "SHOT_MOTION_ZOOM", 1.08)
    monkeypatch.setattr(config, "SHOT_MOTION_HOOK_ZOOM", 1.18)
    monkeypatch.setattr(config, "IMAGE_ZOOM", 1.12)


def planned(kinds: str, **overrides) -> list[Shot]:
    """A settled shot list from a string of kinds: "vvi" is clip, clip, still."""
    shots = [
        shot(index=i, beat_index=i // 2, asset_index=i, start=i * 3.0,
             kind="image" if letter == "i" else "video",
             path=Path("/nowhere/x.jpg" if letter == "i" else "/nowhere/c.mp4"),
             **overrides)
        for i, letter in enumerate(kinds)
    ]
    assemble._plan_motion(shots)
    return shots


def test_the_opening_shot_pushes_in_harder_than_the_rest(moving):
    shots = planned("vvv")

    assert (shots[0].motion, shots[0].motion_zoom) == ("push", 1.18)
    assert shots[1].motion_zoom == 1.08


def test_the_opening_shot_never_pulls_back(moving):
    """A pull-back on the first shot moves away from the subject at exactly the
    moment the reel is asking to be watched."""
    assert planned("v")[0].motion == "push"


def test_clips_push_and_pull_in_turn(moving):
    """A run of shots all moving the same way reads as one mechanical move on
    repeat, which is the thing being fixed rather than a milder version of it."""
    assert [s.motion for s in planned("vvvvv")] == [
        "push", "pull", "push", "pull", "push"]


def test_push_mode_moves_every_shot_the_same_way(moving, monkeypatch):
    monkeypatch.setattr(config, "SHOT_MOTION", "push")

    assert {s.motion for s in planned("vvvvv")} == {"push"}


def test_motion_off_leaves_clips_exactly_as_they_were(moving, monkeypatch):
    monkeypatch.setattr(config, "SHOT_MOTION", "off")
    shots = planned("vvv")

    assert [s.motion for s in shots] == ["", "", ""]
    assert {s.motion_zoom for s in shots} == {1.0}


def test_a_zoom_of_one_is_no_motion_at_all(moving, monkeypatch):
    monkeypatch.setattr(config, "SHOT_MOTION_ZOOM", 1.0)

    assert [s.motion for s in planned("vvv")] == ["", "", ""]


def test_stills_keep_their_own_zoom_and_always_push(moving):
    """IMAGE_ZOOM governs stills and SHOT_MOTION governs clips, because a still
    has no motion of its own at all and a clip only needs enough to read as a
    camera move."""
    shots = planned("vivi")

    assert [(s.motion, s.motion_zoom) for s in shots[1::2]] == [
        ("push", 1.12), ("push", 1.12)]


def test_turning_clip_motion_off_does_not_freeze_the_stills(moving, monkeypatch):
    monkeypatch.setattr(config, "SHOT_MOTION", "off")
    shots = planned("vi")

    assert shots[0].motion == ""
    assert (shots[1].motion, shots[1].motion_zoom) == ("push", 1.12)


def test_a_still_with_its_zoom_switched_off_does_not_move(moving, monkeypatch):
    monkeypatch.setattr(config, "IMAGE_ZOOM", 1.0)
    shots = planned("vi")

    assert shots[1].motion == ""
    assert shots[0].motion, "the clip is unaffected"


def test_a_still_takes_its_turn_in_the_alternation(moving):
    """It does not honour the pull half, but it does not skip the count either -
    so the clips either side carry on in the order they would have had."""
    assert [s.motion for s in planned("vivv")] == ["push", "push", "push", "pull"]


def test_motion_is_planned_after_merging_so_the_alternation_has_no_gaps(moving):
    """Two neighbours showing the same thing are folded into one shot, and the
    fold renumbers everything after it. Deciding direction before that would
    leave two shots running the same way where the seam was."""
    # A still used twice running restarts on the frame it just left, so these
    # two are one shot by the time anything asks how they should move.
    raw = [
        shot(index=0, asset_index=0, start=0.0, kind="video"),
        shot(index=1, asset_index=1, start=3.0, kind="image",
             path=Path("/nowhere/x.jpg")),
        shot(index=2, asset_index=1, start=6.0, kind="image",
             path=Path("/nowhere/x.jpg")),
        shot(index=3, asset_index=2, start=9.0, kind="video"),
        shot(index=4, asset_index=3, start=12.0, kind="video"),
    ]
    merged = assemble._merge_identical_neighbours(raw)
    assemble._plan_motion(merged)

    assert len(merged) == 4, "the repeated still folded into one shot"
    assert [s.index for s in merged] == [0, 1, 2, 3]
    # Index 3 is the second clip after the fold. Had direction been read off the
    # pre-merge numbering it would still be sitting on 4, and pushing like 2.
    assert [s.motion for s in merged] == ["push", "push", "push", "pull"]


def test_a_timeline_off_the_planner_comes_out_already_moving(moving):
    shots, _ = plan(make_manifest(beats=3, assets_per_beat=1))

    assert [s.index for s in shots] == list(range(len(shots)))
    assert all(s.motion for s in shots), "every shot was given a move"
    assert shots[0].motion_zoom == config.SHOT_MOTION_HOOK_ZOOM


def test_the_manifest_records_how_every_shot_moved(moving):
    shots = planned("vvi")
    entries = [
        {"motion": s.motion, "motion_zoom": s.motion_zoom} for s in shots
    ]

    assert entries == [
        {"motion": "push", "motion_zoom": 1.18},
        {"motion": "pull", "motion_zoom": 1.08},
        {"motion": "push", "motion_zoom": 1.12},
    ]


# --- and what that becomes in the filter graph ------------------------------

def test_a_moving_clip_settles_its_frame_rate_before_zoompan_counts_frames():
    """zoompan with d=1 emits one output frame per *input* frame, so a 24fps
    source left unnormalised hands back 24 frames labelled as 30 and the shot
    plays a quarter too fast. Measured on real 24fps and 30fps sources, this
    ordering returns the identical frame count to the static path."""
    chain = assemble._shot_chain(
        shot(motion="push", motion_zoom=1.08), "0:v", "v0")

    assert chain.index(f"fps={config.VIDEO_FPS}") < chain.index("zoompan")


def test_a_moving_clip_emits_one_frame_per_input_frame():
    chain = assemble._shot_chain(
        shot(motion="push", motion_zoom=1.08), "0:v", "v0")

    assert ":d=1:" in chain


def test_a_moving_still_builds_the_whole_shot_out_of_its_one_frame():
    chain = assemble._shot_chain(
        shot(kind="image", path=Path("/nowhere/x.jpg"), duration=2.0,
             motion="push", motion_zoom=1.12), "0:v", "v0")

    assert f":d={2 * config.VIDEO_FPS}:" in chain
    assert "fps=" in chain, "zoompan still needs its own output rate"


def test_a_push_starts_on_the_frame_and_a_pull_ends_on_it():
    push = assemble._shot_chain(shot(motion="push", motion_zoom=1.08), "0:v", "v0")
    pull = assemble._shot_chain(shot(motion="pull", motion_zoom=1.08), "0:v", "v0")

    assert "z='min(1+0.0800*on/" in push
    assert "z='max(1.0800-0.0800*on/" in pull


def test_the_zoom_is_clamped_at_both_ends():
    """`on` keeps counting past the frame the shot was planned for when a source
    hands over more frames than expected; an unclamped ramp zooms into it."""
    push = assemble._shot_chain(shot(motion="push", motion_zoom=1.08), "0:v", "v0")
    pull = assemble._shot_chain(shot(motion="pull", motion_zoom=1.08), "0:v", "v0")

    assert ",1.0800)'" in push
    assert ",1.0000)'" in pull


def test_a_moving_shot_is_oversampled_before_it_is_moved():
    """zoompan quantises its zoom to whole input pixels, so the move is only as
    smooth as the picture it crops out of."""
    chain = assemble._shot_chain(shot(motion="push", motion_zoom=1.08), "0:v", "v0")
    big_w = int(config.VIDEO_WIDTH * assemble.MOTION_OVERSAMPLE) // 2 * 2

    assert f"crop={big_w}:" in chain
    assert f"s={config.VIDEO_WIDTH}x{config.VIDEO_HEIGHT}" in chain, \
        "zoompan lands it on the output size itself"


def test_a_static_shot_still_crops_straight_to_the_frame():
    chain = assemble._shot_chain(shot(), "0:v", "v0")

    assert f"crop={config.VIDEO_WIDTH}:{config.VIDEO_HEIGHT}" in chain
    assert "zoompan" not in chain
def test_a_beat_splits_rather_than_holding_past_the_curve(monkeypatch):
    """The measured fault: a window between one and one-and-a-half targets
    rounded down to a single shot and then held for all of it.

    4.5s against a 3.2s target is 1.4x - it used to round to one shot and hold
    4.5s while the beat's second downloaded clip went unused.
    """
    monkeypatch.setattr(config, "SHOT_PACING", "flat")
    monkeypatch.setattr(config, "SHOT_TARGET_SECONDS", 3.2)
    monkeypatch.setattr(config, "SHOT_MAX_OVERRUN", 1.25)

    assert assemble._shot_count(4.5, 2, position=0.0, total=40.0) == 2


def test_the_overrun_ceiling_can_be_switched_off(monkeypatch):
    monkeypatch.setattr(config, "SHOT_PACING", "flat")
    monkeypatch.setattr(config, "SHOT_TARGET_SECONDS", 3.2)
    monkeypatch.setattr(config, "SHOT_MAX_OVERRUN", 0.0)

    assert assemble._shot_count(4.5, 2, position=0.0, total=40.0) == 1


def test_the_overrun_ceiling_never_invents_footage(monkeypatch):
    """It is still capped at twice what step 4 sourced for the beat, so a beat
    with one clip cannot be cut into four."""
    monkeypatch.setattr(config, "SHOT_PACING", "flat")
    monkeypatch.setattr(config, "SHOT_TARGET_SECONDS", 3.2)
    monkeypatch.setattr(config, "SHOT_MAX_OVERRUN", 1.25)

    assert assemble._shot_count(20.0, 1, position=0.0, total=40.0) <= 2


def test_a_window_already_inside_the_curve_is_left_alone(monkeypatch):
    monkeypatch.setattr(config, "SHOT_PACING", "flat")
    monkeypatch.setattr(config, "SHOT_TARGET_SECONDS", 3.2)
    monkeypatch.setattr(config, "SHOT_MAX_OVERRUN", 1.25)

    assert assemble._shot_count(3.0, 3, position=0.0, total=40.0) == 1


def test_the_ceiling_follows_the_curve_rather_than_a_constant(monkeypatch):
    """The point of measuring the overrun against the curve: the same window
    is a long hold at the top of a reel and an ordinary shot in the middle."""
    monkeypatch.setattr(config, "SHOT_PACING", "curve")
    monkeypatch.setattr(config, "SHOT_HOOK_SECONDS", 1.4)
    monkeypatch.setattr(config, "SHOT_TARGET_SECONDS", 3.2)
    monkeypatch.setattr(config, "SHOT_PACING_RAMP_SECONDS", 8.0)
    monkeypatch.setattr(config, "SHOT_OUTRO_WINDOW_SECONDS", 0.0)
    monkeypatch.setattr(config, "SHOT_MAX_OVERRUN", 1.25)

    opening = assemble._shot_count(3.0, 4, position=0.0, total=40.0)
    middle = assemble._shot_count(3.0, 4, position=20.0, total=40.0)
    assert opening > middle





def test_a_motion_trial_can_freeze_the_shots_for_one_reel(moving):
    """`motion` overrides SHOT_MOTION for this reel only.

    Every reel in the ledger answered the motion question "on", so nothing in
    it could say whether the move earns the crop it costs. A trial has to be
    able to turn it off for one reel without changing the setting for all of
    them.
    """
    shots = [shot(index=i, asset_index=i, start=i * 3.0) for i in range(3)]

    assemble._plan_motion(shots, "off")

    assert [s.motion for s in shots] == ["", "", ""]
    assert all(s.motion_zoom == 1.0 for s in shots)


def test_no_override_leaves_the_setting_in_charge(moving):
    shots = [shot(index=i, asset_index=i, start=i * 3.0) for i in range(3)]

    assemble._plan_motion(shots)

    assert [s.motion for s in shots] == ["push", "pull", "push"]
