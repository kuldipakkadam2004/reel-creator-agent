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


def plan(manifest: dict, ok: set[int] | None = None, narration: float = NARRATION):
    return assemble.plan_timeline(manifest, narration, probes_for(manifest, ok))


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


def test_shot_lengths_respect_the_configured_floor_and_ceiling():
    manifest = make_manifest(assets_per_beat=4, beats=3)
    shots, _ = plan(manifest)

    for shot in shots[:-1]:  # the last one absorbs the rounding drift
        assert shot.duration >= config.SHOT_MIN_SECONDS - 0.01
        assert shot.duration <= config.SHOT_MAX_SECONDS + 0.01


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


def test_a_cue_too_wide_for_the_frame_is_scaled_down():
    wide = Cue(words_from([("SUPERCALIFRAGILISTICEXPIALIDOCIOUS", 0.0, 1.0)]),
               0.0, 1.0)
    assert "\\fs" in assemble.build_ass([wide], "Arial Black", None)

    narrow = Cue(words_from([("SHORT", 0.0, 1.0)]), 0.0, 1.0)
    assert "\\fs" not in assemble.build_ass([narrow], "Arial Black", None)


def test_colours_convert_to_ass_bgr_order():
    assert assemble._ass_colour("#FF0000") == "&H000000FF"   # red
    assert assemble._ass_colour("#00FF00") == "&H0000FF00"   # green
    assert assemble._ass_colour("#0000FF") == "&H00FF0000"   # blue
    assert assemble._ass_colour("#FFF") == "&H00FFFFFF"


def test_a_bad_colour_is_rejected_by_name():
    with pytest.raises(assemble.AssemblyError, match="not a #RRGGBB"):
        assemble._ass_colour("cornflower")


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


def test_an_in_point_becomes_a_seek_before_the_input():
    args = assemble._shot_inputs(shot(source_start=2.5))
    assert args.index("-ss") < args.index("-i")
    assert args[args.index("-ss") + 1] == "2.500"


def test_a_still_is_fed_as_one_frame_when_it_will_be_zoomed(monkeypatch):
    monkeypatch.setattr(config, "IMAGE_ZOOM", 1.12)
    args = assemble._shot_inputs(shot(kind="image", path=Path("/nowhere/x.jpg")))
    # -loop 1 here would multiply against zoompan's d= and emit minutes of video
    assert "-loop" not in args


def test_a_still_is_looped_when_the_zoom_is_switched_off(monkeypatch):
    monkeypatch.setattr(config, "IMAGE_ZOOM", 1.0)
    args = assemble._shot_inputs(shot(kind="image", path=Path("/nowhere/x.jpg")))
    assert "-loop" in args and "-t" in args


def test_every_shot_is_cropped_to_the_configured_frame():
    chain = assemble._shot_chain(shot(), "0:v", "v0")
    assert f"crop={config.VIDEO_WIDTH}:{config.VIDEO_HEIGHT}" in chain
    assert "setsar=1" in chain


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
