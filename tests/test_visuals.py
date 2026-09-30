"""Tests for pipeline/visuals.py.

Everything here is offline: providers are stubbed, HTTP is faked, and the
"downloads" are real-but-tiny JPEGs built at runtime, so the suite never calls a
stock API, spends a Gemini token, or touches the real cache directory.

For a live check run `python pipeline/visuals.py "..." --duration 30` by hand.

    python -m pytest tests/test_visuals.py -v
    python tests/test_visuals.py          # works without pytest installed
"""

from __future__ import annotations

import io
import json
import re
import sys
from hashlib import sha1
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

import config  # noqa: E402
from pipeline import pacing  # noqa: E402
from pipeline import visuals  # noqa: E402
from pipeline.visuals import Beat, Candidate  # noqa: E402


SCRIPT = (
    "The ocean is salty because rain is slightly acidic. "
    "For two billion years, rain has been dissolving rock, carrying sodium and "
    "chloride down every river on Earth. The water evaporates. The salt stays. "
    "Every drop that reaches the sea leaves its minerals behind, forever. "
    "That is why the Dead Sea is nine times saltier than the Atlantic."
)


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

def jpeg_bytes(width: int, height: int, tint: int = 30) -> bytes:
    """A real (tiny) JPEG of the requested size, so Pillow can measure it."""
    from PIL import Image

    buffer = io.BytesIO()
    Image.new("RGB", (width, height), (tint % 256, 60, 90)).save(
        buffer, format="JPEG", quality=30
    )
    return buffer.getvalue()


def unique_media(url: str) -> FakeResponse:
    """Distinct bytes per URL - two stock clips are not byte-identical, and
    pretending otherwise would collide with the duplicate detection on test."""
    seed = int(sha1(url.encode()).hexdigest()[:4], 16)
    if url.endswith(".jpg"):
        return FakeResponse(200, body=jpeg_bytes(1080, 1920, tint=seed),
                            headers={"Content-Type": "image/jpeg"})
    return FakeResponse(200, body=f"fake-mp4-{url}-".encode() * 64,
                        headers={"Content-Type": "video/mp4"})


class FakeResponse:
    """Stand-in for requests.Response covering what visuals.py touches."""

    def __init__(self, status_code=200, payload=None, body=b"", headers=None):
        self.status_code = status_code
        self._payload = payload
        self.content = body if payload is None else json.dumps(payload).encode()
        self.headers = dict(headers or {})

    @property
    def text(self):
        return self.content.decode("utf-8", "replace")

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload

    def iter_content(self, chunk_size=8192):
        for start in range(0, len(self.content), chunk_size):
            yield self.content[start:start + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class FakeSession:
    """Routes GETs by URL substring. Records every call for assertions."""

    def __init__(self, routes=None, default=None):
        self.headers = {}
        self.routes = dict(routes or {})
        self.default = default
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None, stream=False):
        self.calls.append({"url": url, "params": params, "stream": stream})
        for fragment, response in self.routes.items():
            if fragment in url:
                return response(url) if callable(response) else response
        if self.default is not None:
            return self.default(url) if callable(self.default) else self.default
        return FakeResponse(404, body=b"no route")

    @property
    def download_calls(self):
        return [c for c in self.calls if c["stream"]]


def make_candidate(index: int, kind="video", width=1080, height=1920, duration=8.0,
                   provider="stub"):
    return Candidate(
        provider=provider,
        provider_id=f"{kind}{index}",
        kind=kind,
        width=width,
        height=height,
        download_url=f"https://stub.test/{kind}/{index}.{'mp4' if kind == 'video' else 'jpg'}",
        source_url=f"https://stub.test/page/{index}",
        creator=f"creator{index}",
        license="Stub License",
        title=f"clip {index}",
        duration=duration if kind == "video" else None,
    )


class StubProvider(visuals.Provider):
    """Keyless provider that serves whatever `catalog` is set to."""

    name = "stub"
    kinds = ("video", "image")
    key_env = ()
    attribution_required = False
    catalog: list[Candidate] = []
    failure: Exception | None = None

    def search(self, query, kind, limit):
        self.searches += 1
        if StubProvider.failure is not None:
            raise StubProvider.failure
        return [c for c in StubProvider.catalog if c.kind == kind][:limit]


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Point every on-disk location at tmp_path and install the stub provider.

    The relevance screener is pinned offline here, and that is not tidying: on
    a machine with a GEMINI_API_KEY in .env it defaults to "llm", so three of
    the CLI tests below were making real model calls - 40 to 49 seconds each,
    real quota, on every run of the suite. It also meant the tests exercised a
    different path depending on whether a key happened to be present, which is
    the one thing a test must never do. Every test that means to screen passes
    `relevance_mode` explicitly and is unaffected.
    """
    monkeypatch.setattr(config, "VISUAL_RELEVANCE_MODE", "lexical")
    monkeypatch.setattr(config, "VISUALS_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "VISUALS_RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "VISUALS_USAGE_FILE", tmp_path / "usage.json")
    monkeypatch.setattr(visuals, "_PROVIDERS", {"stub": StubProvider})

    # 40 clips served per search, not 15. The stub ignores the query, so every
    # beat sees the same page and a clip one beat takes is gone for all the
    # others - which makes the whole sandbox world SEARCH_PAGE_SIZE clips wide,
    # where a real run gives each beat its own pool of about 30. At 15 the
    # default 8-beat script was already spending the world down to nothing on
    # its last beat, so a test about manifest shape would fail the moment any
    # beat asked for a third clip. Widening it puts the capacity cliff out of
    # the way of the tests that are not about capacity; the ones that are set
    # their own catalog.
    monkeypatch.setattr(visuals, "SEARCH_PAGE_SIZE", 40)
    StubProvider.catalog = [make_candidate(i) for i in range(40)]
    StubProvider.failure = None

    session = FakeSession(default=unique_media)
    monkeypatch.setattr(visuals, "_new_session", lambda: session)
    yield session
    StubProvider.catalog = []
    StubProvider.failure = None


@pytest.fixture
def one_shot_beats(monkeypatch):
    """Pin every beat to a single shot, so it asks for `assets_per_beat` flat.

    The tests below are about the headroom top-up and the acquisition order,
    and they want the beats themselves to source as little as possible so that
    what they are watching is the only thing that can add footage.
    `assets_per_beat=1` used to say that by itself; now a beat also asks for a
    clip per shot it will be cut into, and on this script the opening beat is
    three shots long. Leaving `Beat.shots` at its default says the old thing.
    """
    monkeypatch.setattr(visuals, "plan_shot_demand", lambda beats, duration: None)


def run_visuals(**kwargs):
    """source_visuals with the arguments every sandbox test wants."""
    params = {
        "script": SCRIPT,
        "duration_seconds": 40.0,
        "topic": "why the ocean is salty",
        "providers": ["stub"],
        "query_mode": "heuristic",
        # Both model-backed passes are pinned off by default: these tests run
        # offline, and the screener has its own tests that stub the call.
        "relevance_mode": "lexical",
    }
    params.update(kwargs)
    script = params.pop("script")
    return visuals.source_visuals(script, **params)


# ---------------------------------------------------------------------------
# Beat segmentation
# ---------------------------------------------------------------------------

def test_beats_cover_the_whole_narration_exactly():
    beats = visuals.segment_script(SCRIPT, 40.0)
    assert beats[0].start == 0.0
    assert beats[-1].end == 40.0
    for earlier, later in zip(beats, beats[1:]):
        assert earlier.end == pytest.approx(later.start)
    assert sum(b.duration for b in beats) == pytest.approx(40.0, abs=0.05)


def test_beat_durations_stay_inside_the_window():
    for total in (20.0, 40.0, 75.0):
        beats = visuals.segment_script(SCRIPT, total)
        assert len(beats) > 1
        for beat in beats:
            # The ceiling is soft: merging a stub beat into a neighbour may add
            # up to one minimum beat's worth. Nothing may fall under the floor.
            ceiling = visuals.BEAT_MAX_SECONDS + visuals.BEAT_MIN_SECONDS
            assert beat.duration <= ceiling + 0.01, beat
            assert beat.duration >= visuals.BEAT_MIN_SECONDS - 0.01, beat


# ---------------------------------------------------------------------------
# How much footage a beat asks for
# ---------------------------------------------------------------------------

def test_a_long_beat_is_planned_for_more_shots_than_a_short_one():
    beats = visuals.segment_script(SCRIPT, 40.0)
    visuals.plan_shot_demand(beats, 40.0)
    assert all(b.shots >= 1 for b in beats)
    longest = max(beats, key=lambda b: b.duration)
    shortest = min(beats, key=lambda b: b.duration)
    assert longest.shots > shortest.shots, (
        "every beat was planned for the same number of shots regardless of "
        "how long it is"
    )


def test_the_ask_matches_what_step_5_will_cut_the_beat_into():
    """The defect this exists for: supply on a curve, demand on a constant.

    Step 5 divides a beat by the pacing curve, so a long beat becomes three
    shots. Step 4 used to source a flat two clips for it either way, and step 5
    filled the third shot by cycling back to the first clip - which is how the
    Crescent City reel closed on the aerial it had opened that same beat with.
    """
    beats = visuals.segment_script(SCRIPT, 40.0)
    visuals.plan_shot_demand(beats, 40.0)
    settings = {"assets_per_beat": 2}
    for beat in beats:
        wanted = visuals._assets_wanted(beat, settings)
        shots = pacing.shot_count(
            beat.duration, wanted,
            position=beat.start, total=40.0 + config.VIDEO_TAIL_SECONDS,
        )
        assert shots <= wanted, (
            f"beat {beat.index} is cut into {shots} shots with only "
            f"{wanted} clips to fill them"
        )


def test_a_one_shot_beat_still_gets_its_alternate():
    """The knob is a floor, not a target: a spare clip is cheap insurance."""
    beat = visuals.Beat(index=0, text="x", start=0.0, end=3.0, shots=1)
    assert visuals._assets_wanted(beat, {"assets_per_beat": 2}) == 2


def test_the_ask_is_capped_at_twice_the_floor():
    """One over-long beat may not spend the run's whole download budget."""
    beat = visuals.Beat(index=0, text="x", start=0.0, end=60.0, shots=9)
    assert visuals._assets_wanted(beat, {"assets_per_beat": 2}) == 4
    assert visuals._assets_wanted(beat, {"assets_per_beat": 1}) == 2


def test_every_beat_is_sourced_for_all_of_its_shots(sandbox):
    manifest = run_visuals()
    assert any(b["shots"] > 1 for b in manifest["beats"]), "nothing to prove"
    for beat in manifest["beats"]:
        assert len(beat["asset_indexes"]) >= min(beat["shots"], 4), beat


def test_longer_audio_means_the_same_words_get_more_time():
    short = visuals.segment_script(SCRIPT, 20.0)
    long = visuals.segment_script(SCRIPT, 60.0)
    assert sum(b.duration for b in short) == pytest.approx(20.0, abs=0.05)
    assert sum(b.duration for b in long) == pytest.approx(60.0, abs=0.05)


def test_every_word_of_the_script_survives_segmentation():
    beats = visuals.segment_script(SCRIPT, 40.0)
    rejoined = " ".join(b.text for b in beats)
    assert rejoined.split() == SCRIPT.split()


def test_abbreviations_do_not_end_a_sentence():
    text = "Dr. Ballard found the wreck. It sat two miles down."
    assert visuals.split_sentences(text) == [
        "Dr. Ballard found the wreck.",
        "It sat two miles down.",
    ]


def test_single_short_sentence_is_one_beat():
    beats = visuals.segment_script("Salt never leaves.", 4.0)
    assert len(beats) == 1
    assert beats[0].start == 0.0 and beats[0].end == 4.0


def test_long_clause_without_punctuation_is_split_by_words():
    sentence = " ".join(["word"] * 120) + "."
    beats = visuals.segment_script(sentence, 60.0)
    assert len(beats) > 1
    ceiling = visuals.BEAT_MAX_SECONDS + visuals.BEAT_MIN_SECONDS
    assert all(b.duration <= ceiling + 0.01 for b in beats)


@pytest.mark.parametrize("bad", [0, -1, -0.5])
def test_non_positive_duration_rejected(bad):
    with pytest.raises(ValueError, match="must be positive"):
        visuals.segment_script(SCRIPT, bad)


def test_empty_script_has_nothing_to_segment():
    with pytest.raises(ValueError, match="no sentences"):
        visuals.segment_script("   ", 10.0)


# ---------------------------------------------------------------------------
# Crop-aware resolution floor
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("width,height,expected", [
    (1920, 1080, (608, 1080)),      # landscape: sides come off
    (3840, 2160, (1215, 2160)),
    (1080, 1920, (1080, 1920)),     # already 9:16: untouched
    (2000, 2000, (1125, 2000)),     # square: top and bottom come off
])
def test_crop_to_aspect(width, height, expected):
    assert visuals.crop_to_aspect(width, height) == expected


#: The floor these tests measure against, pinned rather than read from
#: `config`. `config.VISUAL_MIN_*` resolves whatever .env the machine happens
#: to carry, so asserting against it made the suite fail whenever the floor was
#: legitimately retuned - it was testing the operator's configuration instead of
#: the function. The function is what has a contract; the setting is a setting.
FRAME_FLOOR = {"min_width": 1080, "min_height": 1920}


@pytest.mark.parametrize("width,height,passes", [
    (1080, 1920, True),      # exactly the frame
    (2160, 3840, True),      # 4K vertical
    (3840, 2160, True),      # 4K landscape survives the crop
    (1920, 1080, False),     # 1080p landscape does not
    (720, 1280, False),      # too small even though the shape is right
    (1000, 1000, False),
])
def test_resolution_floor_is_measured_after_the_crop(width, height, passes):
    assert visuals.meets_resolution_floor(
        width, height, **FRAME_FLOOR
    ) is passes


def test_floor_can_be_overridden_per_call():
    assert not visuals.meets_resolution_floor(720, 1280, **FRAME_FLOOR)
    assert visuals.meets_resolution_floor(720, 1280, min_width=720, min_height=1280)


@pytest.mark.parametrize("bad", [(0, 1920), (1080, 0), (-5, 10)])
def test_zero_dimensions_never_pass(bad):
    assert visuals.crop_to_aspect(*bad) == (0, 0)
    assert not visuals.meets_resolution_floor(*bad)


# ---------------------------------------------------------------------------
# Candidate filtering and ranking
# ---------------------------------------------------------------------------

def _settings(**overrides):
    base = {
        "kinds": ["video", "image"],
        "allow_landscape": True,
        "min_width": config.VISUAL_MIN_WIDTH,
        "min_height": config.VISUAL_MIN_HEIGHT,
    }
    base.update(overrides)
    return base


def test_usable_candidate_is_not_rejected():
    assert visuals.reject_reason(make_candidate(1), _settings()) == ""


def test_below_floor_is_rejected_with_the_crop_in_the_message():
    """The message has to name the crop, not the source size: "1920x1080" looks
    like plenty and 608x1080 is the number that explains the rejection. Pinned
    to the frame floor so this tests the message rather than today's setting."""
    candidate = make_candidate(1, width=1920, height=1080)
    reason = visuals.reject_reason(
        candidate, _settings(min_width=1080, min_height=1920)
    )
    assert "608x1080" in reason and "floor" in reason


def test_landscape_can_be_refused_outright():
    candidate = make_candidate(1, width=3840, height=2160)
    assert visuals.reject_reason(candidate, _settings()) == ""
    reason = visuals.reject_reason(candidate, _settings(allow_landscape=False))
    assert "landscape" in reason


def test_images_are_rejected_when_disabled():
    candidate = make_candidate(1, kind="image")
    assert "not enabled" in visuals.reject_reason(candidate, _settings(kinds=["video"]))


def test_missing_url_or_dimensions_rejected():
    from dataclasses import replace

    assert "no download url" in visuals.reject_reason(
        replace(make_candidate(1), download_url=""), _settings()
    )
    assert "no dimensions" in visuals.reject_reason(
        replace(make_candidate(1), width=0), _settings()
    )


def _beat(duration=5.0):
    return Beat(index=0, text="the sea", start=0.0, end=duration)


def test_portrait_outranks_landscape():
    portrait = make_candidate(1, width=1080, height=1920)
    landscape = make_candidate(2, width=3840, height=2160)
    assert visuals.score_candidate(portrait, _beat(), provider_rank=0, recently_used=False) \
        > visuals.score_candidate(landscape, _beat(), provider_rank=0, recently_used=False)


def test_video_outranks_a_still_of_the_same_shape():
    video = make_candidate(1, kind="video")
    image = make_candidate(1, kind="image")
    assert visuals.score_candidate(video, _beat(), provider_rank=0, recently_used=False) \
        > visuals.score_candidate(image, _beat(), provider_rank=0, recently_used=False)


def test_clip_that_covers_the_beat_outranks_one_that_does_not():
    covers = make_candidate(1, duration=6.0)
    short = make_candidate(2, duration=1.0)
    assert visuals.score_candidate(covers, _beat(5.0), provider_rank=0, recently_used=False) \
        > visuals.score_candidate(short, _beat(5.0), provider_rank=0, recently_used=False)


def test_recent_use_and_provider_order_push_a_candidate_down():
    candidate = make_candidate(1)
    fresh = visuals.score_candidate(candidate, _beat(), provider_rank=0, recently_used=False)
    stale = visuals.score_candidate(candidate, _beat(), provider_rank=0, recently_used=True)
    second_choice = visuals.score_candidate(
        candidate, _beat(), provider_rank=2, recently_used=False
    )
    assert stale < fresh and second_choice < fresh


# ---------------------------------------------------------------------------
# Relevance - does the clip show what the beat is about?
#
# The cases here are the real ones from the run that prompted this check: a
# black bear returned for "Hadza hunter gatherers", and grey-headed flying
# foxes returned for "brown bat sleeping". The first must be caught. The
# second must NOT be, and is the reason the lexical check is comparative.
# ---------------------------------------------------------------------------

def _described(index: int, slug: str, title: str = "") -> Candidate:
    """A candidate whose only description is the one a provider would give."""
    return Candidate(
        provider="stub", provider_id=f"v{index}", kind="video",
        width=1080, height=1920, duration=8.0,
        download_url=f"https://stub.test/video/{index}.mp4",
        source_url=f"https://www.pexels.com/video/{slug}-{2000 + index}/",
        title=title,
    )


HADZA = Beat(
    index=2, start=0.0, end=6.0, query="Hadza hunter gatherers",
    text="The Hadza hunter-gatherers of Tanzania average six hours a night.",
)
BATS = Beat(
    index=1, start=0.0, end=6.0, query="brown bat sleeping",
    text="Brown bats sleep for nineteen hours a day.",
)
SPERM_WHALE = Beat(
    index=2, start=0.0, end=6.0, query="sperm whale hunting deep",
    text="Except they have one predator that finds them every single day.",
)


def test_description_comes_from_the_url_when_the_title_is_empty():
    # Pexels returns an empty alt for most videos and puts the description in
    # the URL, which is why the slug is read at all.
    candidate = _described(1, "a-black-bear-is-walking-through-a-field")
    assert visuals.describe_candidate(candidate) == "a black bear is walking through a field"


def test_description_uses_both_the_title_and_the_slug():
    candidate = _described(2, "majestic-gorilla-walking", title="Gorilla in a forest")
    described = visuals.describe_candidate(candidate)
    assert "gorilla in a forest" in described and "majestic" in described


def test_provider_chrome_and_the_id_are_not_part_of_the_description():
    # Stopwords are left alone here - they are dropped downstream, where the
    # words are compared. This step only removes what the library adds.
    candidate = _described(3, "free-stock-video-of-a-river")
    described = visuals.describe_candidate(candidate)
    assert "river" in described
    assert not {"free", "stock", "video", "2003"} & set(described.split())


def test_a_clip_sharing_nothing_with_the_beat_scores_zero():
    bear = _described(4, "a-black-bear-is-walking-through-a-field")
    assert visuals.lexical_relevance(bear, HADZA) == 0.0


def test_a_clip_sharing_a_query_word_scores_above_the_floor():
    village = _described(5, "traditional-african-village-gathering-and-dance")
    assert visuals.lexical_relevance(village, HADZA) >= config.VISUAL_RELEVANCE_FLOOR


def test_word_endings_do_not_defeat_the_match():
    # "sleeping" in the query, "sleep" in the description.
    clip = _described(6, "a-man-asleep-in-bed-sleep-study")
    assert visuals.lexical_relevance(clip, BATS) > 0.0


def test_a_plural_reaches_its_own_singular():
    # The giant squid reel of 2026-09-19: beat 2 asked for "sperm whale hunting
    # deep" and Pexels answered "whales on sea" - the right animal, named in the
    # query - and it was recorded at relevance 0.0. "es" was coming off
    # unconditionally, so "whales" stemmed to "whal" while "whale" kept its "e",
    # and `_matches` only tolerates a described word that is LONGER than the
    # term. Every root ending in "e" was unreachable from its own plural.
    whales = _described(60, "whales-on-sea")
    assert visuals.lexical_relevance(whales, SPERM_WHALE) > 0.0


def test_the_es_plural_still_comes_off_a_sibilant():
    # The fix must not overshoot: after s/x/z/ch/sh the "es" is a real ending.
    assert visuals._stem("boxes") == visuals._stem("box")
    assert visuals._stem("beaches") == visuals._stem("beach")
    assert visuals._stem("dishes") == visuals._stem("dish")


def test_short_words_are_not_collapsed_onto_their_neighbours():
    # A trailing "e" comes off "ripple" so that "rippling" can reach it, but
    # doing the same to 4-letter words would merge shots this niche tells
    # apart: a forest fire is not a fir, and a pine is not a pin.
    assert visuals._stem("ripple") == visuals._stem("rippling") == visuals._stem("ripples")
    for pair in (("fire", "fir"), ("pine", "pin"), ("care", "car")):
        assert visuals._stem(pair[0]) != visuals._stem(pair[1]), pair


def test_a_clip_with_no_description_at_all_is_not_judged():
    # Openverse and Pixabay can return neither a title nor a usable slug.
    blank = Candidate(provider="stub", provider_id="x", kind="video",
                      width=1080, height=1920, download_url="https://stub.test/x.mp4")
    assert visuals.lexical_relevance(blank, HADZA) == 1.0


def test_the_lexical_screen_drops_the_bear_when_something_better_exists():
    ranked = [
        (9.0, _described(7, "a-black-bear-is-walking-through-a-field"), "q"),
        (8.5, _described(8, "traditional-african-village-gathering-and-dance"), "q"),
    ]
    verdicts = visuals.screen_lexically(ranked, HADZA, config.VISUAL_RELEVANCE_FLOOR)
    assert verdicts[0][0] is False and verdicts[1][0] is True


def test_the_lexical_screen_keeps_everything_when_nothing_matches_the_wording():
    # Both are right for a beat about bats; neither shares a word with it. A
    # floor applied absolutely would throw the whole beat away.
    ranked = [
        (9.0, _described(9, "grey-headed-flying-foxes-are-amazing"), "q"),
        (8.5, _described(10, "fruit-bat-colony-at-dusk"), "q"),
    ]
    assert visuals.screen_lexically(ranked, BATS, 0.9) == {}


def test_rejects_are_demoted_below_keepers_not_discarded():
    rows = [(9.0, _described(i, f"clip-{i}"), "q") for i in range(4)]
    verdicts = {0: (False, "wrong subject"), 1: (True, ""), 2: (False, "wrong"), 3: (True, "")}
    reordered, notes = visuals._demote_rejects(rows, verdicts)
    assert [r[1].provider_id for r in reordered] == ["v1", "v3", "v0", "v2"]
    assert len(reordered) == len(rows)      # nothing lost
    assert len(notes) == 2 and "wrong subject" in notes[0]


def test_screen_json_survives_fences_and_junk_entries():
    raw = """```json
    [{"beat": 0, "verdicts": [{"i": 0, "ok": true},
                              {"i": 1, "ok": false, "why": "a bear"}]},
     "not an object",
     {"beat": "x", "verdicts": []}]
    ```"""
    parsed, subjects = visuals._parse_screen_json(raw)
    assert parsed == {0: {0: (True, ""), 1: (False, "a bear")}}
    # No "subj" in the response at all: every flag defaults to False, so a
    # model that ignores the new field can never trigger the salvage step.
    assert subjects == {0: {0: False, 1: False}}


def test_screen_json_reads_subj_apart_from_ok():
    """A reject can still be the subject - that pair is what salvage runs on."""
    raw = ('[{"beat": 0, "verdicts": ['
           '{"i": 0, "ok": false, "subj": true, "why": "does not show a wallow"},'
           '{"i": 1, "ok": false, "subj": false, "why": "shows a gaur"}]}]')
    parsed, subjects = visuals._parse_screen_json(raw)
    assert parsed[0][0] == (False, "does not show a wallow")
    assert subjects[0] == {0: True, 1: False}


def test_screen_json_rejects_a_non_array():
    with pytest.raises(ValueError):
        visuals._parse_screen_json('{"beat": 0}')


def test_llm_screen_without_a_key_warns_and_defers_to_lexical(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(config, "GEMINI_API_KEY", "")
    verdicts, warnings = visuals.llm_screen(
        {2: [_described(11, "a-black-bear-is-walking-through-a-field")]},
        [HADZA], "sleep",
    )
    assert verdicts == {}
    assert warnings and "lexical" in warnings[0]


def test_the_screen_draws_the_line_at_a_different_culture_not_a_different_century():
    """Three rules here pull against each other and all three are load bearing.

    Rejecting another civilisation's artefact is the point - an Indian temple
    under narration about Inca stonework says something false about a real
    people.

    The century rule is deliberately *graded* rather than either amnesty. An
    early draft rejected every modern stand-in and left three beats of a
    ten-beat reel with nothing at all. The draft that replaced it kept every
    modern stand-in - "runners over narration about messengers... Keep them" -
    and that is what put an athlete in a jersey on a painted running track
    under "messengers who sprinted a hundred fifty miles a day", a cement mixer
    under "they built roads with stone tools", and a motorway with lane
    markings under "without the wheel". Measured on the Inca run: the three
    beats that shipped the worst footage came back from the screen with an
    empty note list, meaning it had approved all of them.

    So the line is what the frame would have to contain: a thing unchanged by
    the centuries is kept, a thing that could only be the present day is not.
    """
    instruction = visuals._SCREEN_SYSTEM_INSTRUCTION
    assert "Greek inscription" in instruction        # the culture rule
    # ...the half that keeps a timeless stand-in...
    assert "would have looked much the same in the period" in instruction
    # ...and the half that stops a modern one.
    assert "could only be the present day" in instruction
    assert "hi-vis" in instruction
    assert "a different named culture or place" in visuals._SCREEN_PROMPT_TEMPLATE
    assert "unchanged by the centuries" in visuals._SCREEN_PROMPT_TEMPLATE


def test_the_screen_prompt_tells_the_model_to_prefer_keeping():
    # The failure mode of a strict screener is an empty beat, which is worse
    # than a loose match, so the instruction to err towards keeping is load
    # bearing rather than decorative.
    prompt = visuals._SCREEN_PROMPT_TEMPLATE
    assert "Prefer keeping" in prompt
    assert "flying fox" in visuals._SCREEN_SYSTEM_INSTRUCTION


def test_the_establishing_rules_reverse_the_lenient_one():
    """The neutral shot is not choosing against a blank screen.

    "Prefer keeping" is right for a beat's own footage, where the alternative
    is nothing. It is wrong for a fallback, where the beat already has
    something and the fallback is only topping it up: staying one clip short
    is an available answer there, so the close calls should go the other way.

    Measured on the bison reel's neutral shot (four runs each, identical
    every run): without these rules 2 of its 3 documented rejects were kept,
    with them 0, and no good clip was lost either way.
    """
    rules = visuals._SCREEN_ESTABLISHING_RULES
    assert "does not apply here" in rules
    assert "Be strict, not lenient, about anything alive" in rules
    # The daisies and the thistles are what needed saying. A wrong animal was
    # already caught by the ordinary rules; a plant reads as scenery.
    assert "an animal, a person or a plant" in rules


def test_unknown_relevance_mode_warns_and_screens_nothing():
    warnings: list[str] = []
    plans = {0: [(9.0, _described(12, "anything"), "q")]}
    notes = visuals._screen_beats(
        plans, [HADZA], "topic", [], {"relevance_mode": "sideways"},
        None, "run", None, {}, warnings,
    )
    assert notes == {}
    assert any("sideways" in w for w in warnings)


def test_a_beat_with_no_match_is_researched_on_its_fallback_query(monkeypatch):
    """The bear's beat, end to end through the screen.

    The primary query returns only the wrong subject, so the beat must go back
    out on its `alt` query rather than settle - which is the whole point of
    noticing, and the step the old count-based gate never reached because two
    well-shot wrong clips satisfied it.
    """
    beat = Beat(
        index=0, start=0.0, end=6.0, query="Hadza hunter gatherers",
        text="The Hadza hunter-gatherers of Tanzania average six hours a night.",
        alt_queries=("african village people",),
    )
    served = {
        "Hadza hunter gatherers": [
            _described(20, "a-black-bear-is-walking-through-a-field"),
        ],
        "african village people": [
            _described(21, "african-village-people-cooking"),
        ],
    }

    class QueryAwareProvider(visuals.Provider):
        name = "stub"
        kinds = ("video",)
        key_env = ()
        attribution_required = False

        def search(self, query, kind, limit):
            return list(served.get(query, []))

    provider = QueryAwareProvider(session=None)
    plans = {0: [(9.0, served["Hadza hunter gatherers"][0], "Hadza hunter gatherers")]}
    warnings: list[str] = []
    settings = _settings(
        relevance_mode="llm",
        relevance_floor=config.VISUAL_RELEVANCE_FLOOR,
        relevance_shortlist=6,
        assets_per_beat=1,
        reuse_cooldown=0,
        kinds=["video"],
    )
    # Stubbed rather than called: the retry is driven by an explicit reject,
    # which is what the model returns for this beat on a live run.
    monkeypatch.setattr(
        visuals, "llm_screen",
        lambda shortlists, beats, topic, subject_keys=None, establishing=False: (
            {0: {0: (False, "shows a bear")}}, []),
    )
    notes = visuals._screen_beats(
        plans, [beat], "sleep", [provider], settings,
        visuals.UsageLedger(), "run", visuals._RunState(), {}, warnings,
    )

    # The village clip found on the fallback query now outranks the bear, and
    # the bear is still there underneath rather than thrown away.
    ordered = [row[1].provider_id for row in plans[0]]
    assert ordered[0] == "v21", ordered
    assert "v20" in ordered
    assert any("re-searched" in note for note in notes[0])


def _query_aware_provider(served):
    """A provider answering each query with exactly what `served` maps it to."""

    class QueryAwareProvider(visuals.Provider):
        name = "stub"
        kinds = ("video",)
        key_env = ()
        attribution_required = False
        asked = []

        def search(self, query, kind, limit):
            QueryAwareProvider.asked.append(query)
            return list(served.get(query, []))

    QueryAwareProvider.asked = []
    return QueryAwareProvider


def _inca_beat():
    return Beat(
        index=0, start=0.0, end=6.0, query="ancient construction workers",
        text="The state conscripted every able-bodied commoner to work.",
        alt_queries=("historical laborers",),
        neutral_query="Andes mountain landscape",
    )


def _retry_settings():
    return _settings(
        relevance_mode="llm",
        relevance_floor=config.VISUAL_RELEVANCE_FLOOR,
        relevance_shortlist=6,
        assets_per_beat=1,
        reuse_cooldown=0,
        kinds=["video"],
    )


def _reject_everything(shortlists, beats, topic, subject_keys=None, establishing=False):
    return (
        {index: {position: (False, "shows modern construction")
                 for position in range(len(candidates))}
         for index, candidates in shortlists.items()},
        [],
    )


def test_the_neutral_fill_names_the_real_reason_it_was_needed():
    """The note used to assert a download failure it had never checked for.

    `filled N slot(s) from the neutral shot ...: what the screen kept could not
    be downloaded` was hard-coded on a branch whose only condition is that the
    beat came up short. On the bedrock reel (2026-09-10) it fired on three
    beats while the run recorded no download failure at all - every real one
    warns, and there were none. The commonest cause is the opposite of what it
    claimed: the screen kept fewer clips than the beat has slots, so nothing
    was ever downloaded and nothing could have failed.

    The four causes want different responses - too few keepers is a supply
    problem, an already-spent key is the cross-beat dedupe working - so the
    note has to tell them apart.
    """
    rows = [(1.0, make_candidate(i), "q") for i in range(3)]
    state = visuals._RunState()

    # Too few keepers: nothing was downloaded, so nothing failed.
    reason = visuals._shortfall_reason(rows[:1], [{}], want=2, state=state)
    assert "kept only 1" in reason
    assert "nothing failed to download" in reason

    # A keeper that really did fail is still reported as one.
    state.failed_keys.add(rows[1][1].key)
    reason = visuals._shortfall_reason(rows[:2], [{}], want=2, state=state)
    assert "failed to download" in reason
    assert "nothing failed" not in reason

    # A keeper an earlier beat already used is not a failure at all.
    state.failed_keys.clear()
    state.taken_keys.add(rows[1][1].key)
    reason = visuals._shortfall_reason(rows[:2], [{}], want=2, state=state)
    assert "already used by an earlier beat" in reason

    # `taken_keys` is run-wide, so a beat's own placement is in it too. The
    # clip this beat just used must not be read back as an earlier beat's.
    state.taken_keys.clear()
    state.taken_keys.add(rows[0][1].key)
    placed = [{"provider_key": rows[0][1].key}]
    reason = visuals._shortfall_reason(rows[:2], placed, want=2, state=state)
    assert "already used by an earlier beat" not in reason, reason


def test_a_beat_with_nothing_honest_falls_back_to_the_neutral_shot(monkeypatch):
    """The cement mixer's beat, end to end.

    Both queries come back showing present-day construction, so there is no
    honest shot to be had. What shipped before this existed was the top-ranked
    reject: a man working a cement mixer under narration about people who built
    with stone tools. A neutral shot of the subject's own landscape says less
    than the beat asked for, but it cannot state something false.
    """
    beat = _inca_beat()
    served = {
        "ancient construction workers": [
            _described(30, "a-man-is-working-on-a-cement-mixer"),
        ],
        "historical laborers": [
            _described(31, "construction-worker-setting-up-scaffolding"),
        ],
        "Andes mountain landscape": [
            _described(32, "snow-covered-andes-peaks-at-dawn"),
        ],
    }
    provider = _query_aware_provider(served)(session=None)
    monkeypatch.setattr(visuals, "llm_screen", _reject_everything)

    plans = {0: [(9.0, served["ancient construction workers"][0],
                  "ancient construction workers")]}
    warnings = []
    notes = visuals._screen_beats(
        plans, [beat], "how the Inca built roads without the wheel", [provider],
        _retry_settings(), visuals.UsageLedger(), "run", visuals._RunState(),
        {}, warnings,
    )

    ordered = [row[1].provider_id for row in plans[0]]
    assert ordered[0] == "v32", f"the neutral shot did not take the beat: {ordered}"
    # The rejects stay underneath rather than being thrown away, so a later
    # stage that runs out of candidates still has something to place.
    assert "v30" in ordered and "v31" in ordered
    assert any("neutral shot" in note for note in notes[0])
    assert any("rather than a contradicting one" in w for w in warnings)


def test_the_neutral_shot_will_not_place_a_clip_the_run_rejected(monkeypatch):
    """The sunflower sea star leak, from the reel of 2026-09-11.

    Two beats, and the aquarium clip is in both pools. Beat 0 puts it in front
    of the screen, which rejects it by name. Beat 1 never ranks it, finds
    nothing it can keep, and falls back to its neutral shot - whose query
    returns that same aquarium. Because the neutral path is deliberately
    unscreened, it used to place the reject with nothing recorded.

    On the shipped reel this happened to four beats out of eight, and beats 2
    and 3 swapped: each shipped the clip the other had just thrown out. A
    verdict on what a clip shows does not expire when the beat changes.
    """
    aquarium = _described(40, "fishes-and-plants-in-aquarium")
    rejected = Beat(
        index=0, start=0.0, end=6.0, query="sunflower sea star hunting",
        text="It glides a meter a minute across the ocean floor.",
    )
    falls_back = Beat(
        index=1, start=6.0, end=12.0, query="purple sea urchin grazing",
        text="Its main course is the purple sea urchin.",
        neutral_query="underwater kelp forest",
    )
    served = {
        "purple sea urchin grazing": [_described(41, "a-stingray-is-swimming")],
        "underwater kelp forest": [aquarium],
    }
    provider = _query_aware_provider(served)(session=None)
    monkeypatch.setattr(visuals, "llm_screen", _reject_everything)

    plans = {
        0: [(9.0, aquarium, "sunflower sea star hunting")],
        1: [(9.0, served["purple sea urchin grazing"][0],
             "purple sea urchin grazing")],
    }
    rejected_keys: set[str] = set()
    warnings = []
    visuals._screen_beats(
        plans, [rejected, falls_back], "why sunflower sea stars matter",
        [provider], _retry_settings(), visuals.UsageLedger(), "run",
        visuals._RunState(), {}, warnings, rejected_keys,
    )

    assert aquarium.key in rejected_keys, "beat 0 did not reject the aquarium"
    placed = [row[1].key for row in plans[1]]
    assert aquarium.key not in placed, (
        "the neutral shot re-placed a clip this run had already rejected: "
        f"{placed}"
    )


def test_the_neutral_shot_still_fills_a_beat_when_nothing_was_rejected(monkeypatch):
    """The guard only ever subtracts, and must not cost the neutral shot the
    beats it was written for - `_neutral_plan` with an empty rejection set
    behaves exactly as it did before."""
    beat = _inca_beat()
    served = {
        "ancient construction workers": [
            _described(50, "a-man-is-working-on-a-cement-mixer"),
        ],
        "historical laborers": [
            _described(51, "construction-worker-setting-up-scaffolding"),
        ],
        "Andes mountain landscape": [
            _described(52, "snow-covered-andes-peaks-at-dawn"),
        ],
    }
    provider = _query_aware_provider(served)(session=None)
    monkeypatch.setattr(visuals, "llm_screen", _reject_everything)

    plans = {0: [(9.0, served["ancient construction workers"][0],
                  "ancient construction workers")]}
    notes = visuals._screen_beats(
        plans, [beat], "how the Inca built roads without the wheel", [provider],
        _retry_settings(), visuals.UsageLedger(), "run", visuals._RunState(),
        {}, [],
    )
    assert [row[1].provider_id for row in plans[0]][0] == "v52"
    assert any("neutral shot" in note for note in notes[0])


def test_the_subject_is_salvaged_before_the_neutral_shot(monkeypatch):
    """The bison reel of 2026-09-17, in miniature.

    Beat 3 asked for "bison wallow with water", turned down three genuine bison
    for *does not show a wallow*, and beats 1, 2, 4 and 6 then filled from a
    neutral prairie that returned sheep, daisies and thistles. The reel was
    graded `off_topic` for showing no bison while four screened bison clips sat
    unused in the run. A clip the screen marked `subj` must now be reached
    before the neutral shot, wherever in the run it was found.
    """
    wallow = Beat(
        index=0, start=0.0, end=6.0, query="bison wallow with water",
        text="Those wallows are not just puddles.",
        alt_queries=("prairie puddle rainwater",),
        neutral_query="American prairie landscape",
    )
    standing = _described(60, "bison-in-the-grassy-field-near-a-road")
    prairie = _described(61, "american-prairie-landscape-at-dawn")
    served = {
        "prairie puddle rainwater": [],
        "American prairie landscape": [prairie],
    }
    provider = _query_aware_provider(served)(session=None)

    def screen(shortlists, beats, topic, subject_keys=None, establishing=False):
        # The live screen's exact verdict on those three clips: not usable for
        # this beat, and unmistakably the subject.
        if subject_keys is not None:
            subject_keys.add(standing.key)
        return ({index: {position: (False, "does not show a wallow")
                         for position in range(len(candidates))}
                 for index, candidates in shortlists.items()}, [])

    monkeypatch.setattr(visuals, "llm_screen", screen)

    plans = {0: [(9.0, standing, "bison wallow with water")]}
    warnings = []
    rejected_keys = set()
    salvaged_keys = {}
    notes = visuals._screen_beats(
        plans, [wallow], "how bison help restore a natural landscape",
        [provider], _retry_settings(), visuals.UsageLedger(), "run",
        visuals._RunState(), {}, warnings, rejected_keys,
        salvaged_keys=salvaged_keys,
    )

    placed = [row[1].provider_id for row in plans[0]]
    assert placed[0] == "v60", (
        f"the beat went to the neutral shot with the subject in hand: {placed}"
    )
    assert any("salvaged" in note for note in notes[0])
    assert any("subject but not the action" in w for w in warnings)
    # It must stop reading as a reject on this beat, or pass 2 places it with
    # the "nothing else could be downloaded" warning - the wrong sentence for
    # a clip chosen deliberately. On this beat only: the run-wide set still
    # holds it, for every other beat that turned it down.
    assert salvaged_keys == {0: {standing.key}}
    assert standing.key in rejected_keys


def test_salvage_never_fires_on_a_clip_the_screen_did_not_call_the_subject(
        monkeypatch):
    """`subj` defaults False, so a model that ignores the field changes nothing.

    This is the guard on the step above: salvage reaching for whatever happens
    to be lying around would put back exactly the footage the screen exists to
    keep out - the gaur, the cows and the horses that beat 6 turned down.
    """
    wallow = Beat(
        index=0, start=0.0, end=6.0, query="bison wallow with water",
        text="Those wallows are not just puddles.",
        alt_queries=("prairie puddle rainwater",),
        neutral_query="American prairie landscape",
    )
    gaur = _described(62, "gaur-grazing-in-woodland-clearing")
    prairie = _described(63, "american-prairie-landscape-at-dawn")
    served = {
        "prairie puddle rainwater": [],
        "American prairie landscape": [prairie],
    }
    provider = _query_aware_provider(served)(session=None)

    # A screen that answers no "subj" at all, exactly as an older model would.
    monkeypatch.setattr(visuals, "llm_screen", _reject_everything)

    plans = {0: [(9.0, gaur, "bison wallow with water")]}
    notes = visuals._screen_beats(
        plans, [wallow], "how bison help restore a natural landscape",
        [provider], _retry_settings(), visuals.UsageLedger(), "run",
        visuals._RunState(), {}, [],
    )

    assert [row[1].provider_id for row in plans[0]][0] == "v63"
    assert any("neutral shot" in note for note in notes[0])
    assert not any("salvaged" in note for note in notes[0])


# ---------------------------------------------------------------------------
# The opening beat opens on the subject
#
# The spider reel of 2026-09-22 opened on "The milk in your fridge has less
# protein than a spider's". Step 4 searched for milk, the screen kept six milk
# clips on the line's own words and marked none of them `subj`, and the first
# frame and the cover were both a latte.
# ---------------------------------------------------------------------------

_SPIDER_TOPIC = "how a jumping spider produces milk"


def _milk_beat(shots=2):
    return Beat(
        index=0, start=0.0, end=3.3, query="pouring milk into glass",
        text="The milk in your fridge has less protein than a spider's.",
        shots=shots,
    )


def _spider_beat():
    return Beat(
        index=1, start=3.3, end=7.5, query="jumping spider with babies",
        text="One species of jumping spider raises its young almost entirely "
             "on milk.",
    )


def _screen_by_key(verdicts_by_key):
    """A screen answering each clip by key: `{key: (ok, subj)}`.

    A clip not named is kept and is not the subject, which is what the filler
    beats of an end-to-end run want.
    """
    def screen(shortlists, beats, topic, subject_keys=None, establishing=False):
        verdicts = {}
        for index, candidates in shortlists.items():
            verdicts[index] = {}
            for position, candidate in enumerate(candidates):
                ok, subj = verdicts_by_key.get(candidate.key, (True, False))
                verdicts[index][position] = (ok, "" if ok else "wrong thing")
                if subj and subject_keys is not None:
                    subject_keys.add(candidate.key)
        return verdicts, []
    return screen


def _screen_opening(monkeypatch, plans, beats, verdicts_by_key):
    monkeypatch.setattr(visuals, "llm_screen", _screen_by_key(verdicts_by_key))
    warnings = []
    notes = visuals._screen_beats(
        plans, beats, _SPIDER_TOPIC, [_query_aware_provider({})(session=None)],
        _retry_settings(), visuals.UsageLedger(), "run", visuals._RunState(),
        {}, warnings,
    )
    return notes, warnings


def test_the_opening_beat_opens_on_the_subject_not_the_comparison(monkeypatch):
    """The spider reel, in miniature: the milk stays, but behind the spider."""
    latte = _described(70, "pouring-milk-for-a-perfect-cafe-latte")
    glasses = _described(71, "pouring-milk-into-black-coffee-glasses")
    pipe = _described(72, "close-up-of-jumping-spider-on-outdoor-pipe")
    branch = _described(73, "a-spider-on-a-branch-with-a-blue-sky")
    leaf = _described(74, "jumping-spider-on-a-green-leaf")
    plans = {
        0: [(14.7, latte, "pouring milk into glass"),
            (14.7, glasses, "pouring milk into glass")],
        1: [(8.7, pipe, "jumping spider with babies"),
            (8.7, branch, "jumping spider with babies"),
            (8.7, leaf, "jumping spider with babies")],
    }
    notes, warnings = _screen_opening(monkeypatch, plans,
                                      [_milk_beat(), _spider_beat()], {
        latte.key: (True, False), glasses.key: (True, False),
        pipe.key: (True, True), branch.key: (True, True),
        leaf.key: (True, True),
    })

    opening = [row[1].provider_id for row in plans[0]]
    # Two slots, filled from what beat 1 has to spare - not from the one clip
    # beat 1 will itself place first.
    assert opening[:2] == ["v73", "v74"], opening
    assert "v72" not in opening[:2]
    # The line's own footage is still there, behind the subject.
    assert opening[2:] == ["v70", "v71"], opening
    assert any("first frame and the cover" in note for note in notes[0])
    assert any("borrowed" in w and "beat 0" in w for w in warnings), warnings


def test_the_first_frame_may_take_a_clip_another_beat_needs(monkeypatch):
    """Only one, though. The first frame is worth another beat's clip; the
    second shot is not, so a beat with nothing to spare gives up exactly one."""
    latte = _described(70, "pouring-milk-for-a-perfect-cafe-latte")
    pipe = _described(72, "close-up-of-jumping-spider-on-outdoor-pipe")
    web = _described(75, "spider-web-with-morning-fog")
    plans = {
        0: [(14.7, latte, "pouring milk into glass")],
        1: [(8.7, pipe, "jumping spider with babies"),
            (8.7, web, "jumping spider with babies")],
    }
    spider = _spider_beat()
    spider.shots = 2      # beat 1 will place both of its clips
    notes, warnings = _screen_opening(monkeypatch, plans, [_milk_beat(), spider], {
        latte.key: (True, False), pipe.key: (True, True),
        web.key: (True, False),
    })

    opening = [row[1].provider_id for row in plans[0]]
    assert opening == ["v72", "v70"], opening
    assert any("borrowed" in w for w in warnings)


def test_the_opening_beats_own_subject_footage_goes_first(monkeypatch):
    """Nothing is borrowed while the beat's own screen kept the subject."""
    latte = _described(70, "pouring-milk-for-a-perfect-cafe-latte")
    own_spider = _described(76, "jumping-spider-beside-a-milk-droplet")
    branch = _described(73, "a-spider-on-a-branch-with-a-blue-sky")
    leaf = _described(74, "jumping-spider-on-a-green-leaf")
    plans = {
        0: [(14.7, latte, "pouring milk into glass"),
            (9.0, own_spider, "pouring milk into glass")],
        1: [(8.7, branch, "jumping spider with babies"),
            (8.7, leaf, "jumping spider with babies")],
    }
    notes, warnings = _screen_opening(
        monkeypatch, plans, [_milk_beat(shots=1), _spider_beat()], {
            latte.key: (True, False), own_spider.key: (True, True),
            branch.key: (True, True), leaf.key: (True, True),
        })

    assert [row[1].provider_id for row in plans[0]] == ["v76", "v70"]
    assert any("moved 1 clip(s)" in note for note in notes[0])
    assert not any("borrowed" in w for w in warnings)


def test_a_run_the_screen_named_no_subject_for_is_left_alone(monkeypatch):
    """`subj` defaults false, and a topic that is a process answers false
    throughout. Neither is evidence the opening is wrong."""
    latte = _described(70, "pouring-milk-for-a-perfect-cafe-latte")
    pipe = _described(72, "close-up-of-jumping-spider-on-outdoor-pipe")
    plans = {
        0: [(14.7, latte, "pouring milk into glass")],
        1: [(8.7, pipe, "jumping spider with babies")],
    }
    notes, warnings = _screen_opening(monkeypatch, plans,
                                      [_milk_beat(), _spider_beat()], {
        latte.key: (True, False), pipe.key: (True, False),
    })

    assert [row[1].provider_id for row in plans[0]] == ["v70"]
    assert not any("first frame" in note for note in notes.get(0, []))
    assert not any("beat 0" in w for w in warnings), warnings


def test_a_rejected_subject_clip_is_never_moved_to_the_front(monkeypatch):
    """A reject put first would still be placed last - pass 2 reads
    `rejected_keys` run-wide - and un-rejecting it here would un-reject it on
    every beat that turned it down. So the opening says it is off-subject
    instead of pretending to have fixed it."""
    latte = _described(70, "pouring-milk-for-a-perfect-cafe-latte")
    hunting = _described(77, "close-up-of-spider-eating-prey")
    leaf_litter = _described(78, "forest-leaf-litter-close-up")
    plans = {
        0: [(14.7, latte, "pouring milk into glass")],
        1: [(8.7, hunting, "jumping spider with babies"),
            (8.7, leaf_litter, "jumping spider with babies")],
    }
    rejected_keys: set[str] = set()
    monkeypatch.setattr(visuals, "llm_screen", _screen_by_key({
        latte.key: (True, False), hunting.key: (False, True),
        leaf_litter.key: (True, False),
    }))
    warnings = []
    visuals._screen_beats(
        plans, [_milk_beat(), _spider_beat()], _SPIDER_TOPIC,
        [_query_aware_provider({})(session=None)], _retry_settings(),
        visuals.UsageLedger(), "run", visuals._RunState(), {}, warnings,
        rejected_keys,
    )

    assert [row[1].provider_id for row in plans[0]] == ["v70"]
    assert hunting.key in rejected_keys
    assert any("beat 0" in w and "approved no clip" in w for w in warnings), warnings


def test_a_clip_salvaged_for_one_beat_stays_rejected_on_the_others(
    sandbox, monkeypatch, one_shot_beats
):
    """The spider reel's beat 5, end to end.

    Beat 3 kept nothing and salvaged every clip the screen had called the
    subject. That used to take them out of the run-wide `rejected_keys`, so
    beat 5 - whose own screen had rejected "close up of spider on dewy web" by
    name - read it as a keeper, placed it the moment its real keepers ran out,
    and said nothing. Here beat 1 salvages, and beat 2 is the one that turned
    `dew` down: when beat 2's keeper fails it must go to its neutral shot, not
    to the clip it rejected.
    """
    opening = _described(80, "jumping-spider-close-up")
    wrong = _described(81, "spider-web-weaving")
    salvaged = _described(82, "spider-eating-prey")
    keeper = _described(83, "spiderlings-on-a-leaf")
    dew = _described(84, "close-up-of-spider-on-dewy-web")
    neutral = _described(85, "forest-undergrowth-close-up")
    served = {"q0": [opening], "q1": [wrong, salvaged], "q2": [keeper, dew],
              "neutral": [neutral]}
    fillers: dict[str, list] = {}

    class PerBeatProvider(visuals.Provider):
        name = "stub"
        kinds = ("video",)
        key_env = ()
        attribution_required = False

        def search(self, query, kind, limit):
            if query in served:
                return list(served[query])
            if not re.fullmatch(r"q\d+", query):
                return []
            number = 90 + int(query[1:])
            return fillers.setdefault(query, [_described(number, f"filler-{number}")])

    def queries(beats, topic, mode):
        for beat in beats:
            beat.query = f"q{beat.index}"
            beat.alt_queries = ()
            beat.neutral_query = "neutral" if beat.index == 2 else ""
        return []

    monkeypatch.setattr(visuals, "_PROVIDERS", {"stub": PerBeatProvider})
    monkeypatch.setattr(visuals, "build_queries", queries)
    monkeypatch.setattr(visuals, "llm_screen", _screen_by_key({
        opening.key: (True, True),
        wrong.key: (False, False), salvaged.key: (False, True),
        keeper.key: (True, False), dew.key: (False, True),
    }))
    sandbox.routes[keeper.download_url] = FakeResponse(404, body=b"gone")

    manifest = run_visuals(relevance_mode="llm", assets_per_beat=1,
                           topic=_SPIDER_TOPIC)

    def placed(beat_index):
        return {manifest["assets"][i]["source_url"]
                for i in manifest["beats"][beat_index]["asset_indexes"]}

    # Every beat sourced something. A provider that raises is swallowed into a
    # warning, and this test would otherwise pass on an empty run.
    assert all(beat["asset_indexes"] for beat in manifest["beats"]), (
        manifest["warnings"]
    )
    assert any("salvaged" in note for note in manifest["beats"][1]["notes"])
    assert salvaged.source_url in placed(1)
    assert dew.source_url not in placed(2), (
        "beat 2 placed a clip its own screen rejected, because beat 1 salvaged it"
    )
    assert neutral.source_url in placed(2)


def test_a_beat_no_provider_answered_still_reaches_the_neutral_shot(monkeypatch):
    """"beat 8 had 0 hits and its 7.0s was absorbed by beat 7."

    Nothing found at all is a different failure from found-and-wrong, and it
    used to be handled by not handling it: the screen skipped any beat with an
    empty shortlist, so the beats with no footage were the only ones that could
    not reach the fallback that exists for having no footage. Their seconds
    went to a neighbour instead, and three of the reels binned to 2026-09-09
    were binned for it.
    """
    beat = _inca_beat()
    served = {
        # The primary query answers nothing at all - not a wrong shot, nothing.
        "historical laborers": [],
        "Andes mountain landscape": [
            _described(36, "snow-covered-andes-peaks-at-dawn"),
        ],
    }
    provider = _query_aware_provider(served)(session=None)

    plans = {0: []}
    warnings = []
    notes = visuals._screen_beats(
        plans, [beat], "how the Inca built roads without the wheel", [provider],
        _retry_settings(), visuals.UsageLedger(), "run", visuals._RunState(),
        {}, warnings,
    )

    assert [row[1].provider_id for row in plans[0]] == ["v36"]
    assert any("neutral shot" in note for note in notes[0])
    # And it says which failure it was: "nothing matches the narration" would
    # send whoever reads this to the screen instead of to the providers.
    assert any("no provider returned anything for" in w for w in warnings)
    assert not any("nothing found matches" in w for w in warnings)


def test_a_beat_nothing_can_fill_says_its_seconds_are_going_elsewhere(monkeypatch):
    """The honest ending when even the neutral shot is empty.

    The beat is still lost - nothing here can invent footage - but the warning
    now says what actually happens to it downstream rather than claiming a
    closest-available shot that does not exist.
    """
    beat = _inca_beat()
    provider = _query_aware_provider({})(session=None)

    plans = {0: []}
    warnings = []
    visuals._screen_beats(
        plans, [beat], "how the Inca built roads without the wheel", [provider],
        _retry_settings(), visuals.UsageLedger(), "run", visuals._RunState(),
        {}, warnings,
    )

    assert plans[0] == []
    assert any("absorbed by a neighbour" in w for w in warnings)


def test_the_neutral_shot_is_not_searched_when_the_fallback_query_works(monkeypatch):
    """It is a last resort, not a second guess.

    The neutral is deliberately under-specified, so reaching for it while a
    real match exists would trade a good shot for a merely safe one.
    """
    beat = _inca_beat()
    served = {
        "ancient construction workers": [
            _described(33, "a-man-is-working-on-a-cement-mixer"),
        ],
        "historical laborers": [
            _described(34, "hands-stacking-a-dry-stone-wall"),
        ],
        "Andes mountain landscape": [
            _described(35, "snow-covered-andes-peaks-at-dawn"),
        ],
    }
    provider_cls = _query_aware_provider(served)
    provider = provider_cls(session=None)

    def screen(shortlists, beats, topic, subject_keys=None, establishing=False):
        # Rejects the mixer, keeps the stone wall: the graded rule the live
        # prompt applies.
        out = {}
        for index, candidates in shortlists.items():
            out[index] = {}
            for position, candidate in enumerate(candidates):
                modern = "cement-mixer" in (candidate.source_url or "")
                out[index][position] = (
                    not modern, "shows modern construction" if modern else "",
                )
        return out, []

    monkeypatch.setattr(visuals, "llm_screen", screen)

    plans = {0: [(9.0, served["ancient construction workers"][0],
                  "ancient construction workers")]}
    visuals._screen_beats(
        plans, [beat], "how the Inca built roads without the wheel", [provider],
        _retry_settings(), visuals.UsageLedger(), "run", visuals._RunState(),
        {}, [],
    )

    assert [row[1].provider_id for row in plans[0]][0] == "v34"
    assert "Andes mountain landscape" not in provider_cls.asked, (
        "spent a search on the neutral shot while a real match existed"
    )


def test_the_retry_is_screened_by_the_model_not_the_lexical_check(monkeypatch):
    """The retry has to be judged by the standard that sent the beat back.

    The defect it exists to catch - footage from the wrong century - is
    invisible to word overlap, which reads "cement mixer" and "stone tools" as
    unrelated in exactly the way it reads a correct shot described in other
    words. And `screen_lexically` cannot reject a whole shortlist by
    construction: it returns no verdicts at all unless something clears the
    floor, and keeps whatever does. Screening the retry with it therefore left
    the neutral fallback below unreachable.
    """
    beat = _inca_beat()
    served = {
        "ancient construction workers": [_described(36, "a-cement-mixer-turning")],
        "historical laborers": [_described(37, "a-forklift-on-a-building-site")],
        "Andes mountain landscape": [_described(38, "andes-peaks")],
    }
    provider = _query_aware_provider(served)(session=None)
    seen_shortlists = []

    def screen(shortlists, beats, topic, subject_keys=None, establishing=False):
        seen_shortlists.append(shortlists)
        return _reject_everything(shortlists, beats, topic, subject_keys)

    monkeypatch.setattr(visuals, "llm_screen", screen)

    plans = {0: [(9.0, served["ancient construction workers"][0],
                  "ancient construction workers")]}
    visuals._screen_beats(
        plans, [beat], "how the Inca built roads without the wheel", [provider],
        _retry_settings(), visuals.UsageLedger(), "run", visuals._RunState(),
        {}, [],
    )

    # Three calls, in order: the shortlist, the re-search, and the neutral
    # shot the re-search fell through to - which is screened on the same
    # standard since 2026-09-12 and is why this is not 2.
    assert len(seen_shortlists) == 3, "the re-searched beat was not screened again"
    retried = [c.provider_id for c in seen_shortlists[1][0]]
    assert retried == ["v37"], retried
    assert [c.provider_id for c in seen_shortlists[2][0]] == ["v38"], (
        "the neutral shot was placed without being screened"
    )
    # Rejected by the stub like everything else, and still first: demotion
    # reorders, and there is nothing here to promote above it.
    assert [row[1].provider_id for row in plans[0]][0] == "v38"


def test_the_screen_names_the_keys_it_rejected(monkeypatch):
    """Demotion alone only reorders, which acquisition is free to walk past.

    On the live Inca run beat 8 demoted five modern deliverymen and then
    shipped two of them: the single clip the screen kept did not survive
    downloading, and the rows underneath it were the rejects. Pass 2 can only
    avoid that if it can tell one from the other, so the screen hands the keys
    back rather than only reordering.
    """
    beat = _inca_beat()
    provider = _query_aware_provider({})(session=None)
    monkeypatch.setattr(visuals, "llm_screen", _reject_everything)

    mixer = _described(40, "a-man-is-working-on-a-cement-mixer")
    plans = {0: [(9.0, mixer, "ancient construction workers")]}
    rejected = set()
    visuals._screen_beats(
        plans, [beat], "how the Inca built roads without the wheel", [provider],
        _retry_settings(), visuals.UsageLedger(), "run", visuals._RunState(),
        {}, [], rejected,
    )
    assert mixer.key in rejected


def _keep_only(provider_ids, rejected_ids=None):
    """A screen that keeps the named candidates and rejects every other one."""

    def screen(shortlists, beats, topic, subject_keys=None, establishing=False):
        verdicts = {}
        for index, candidates in shortlists.items():
            verdicts[index] = {}
            for position, candidate in enumerate(candidates):
                ok = candidate.provider_id in provider_ids
                if not ok and rejected_ids is not None:
                    rejected_ids.add(candidate.provider_id)
                verdicts[index][position] = (ok, "" if ok else "modern machinery")
        return verdicts, []

    return screen


def test_a_rejected_clip_is_not_used_while_a_keeper_is_left(
    sandbox, monkeypatch, one_shot_beats
):
    """Acquisition walks down the ranking; the screen only reordered it.

    So a keeper that 404s used to hand the beat straight to the clip below it,
    which is the one the screen had just thrown out. This is the live beat 8:
    five modern deliverymen demoted, and two of them on screen anyway because
    the single keeper above them did not come back.
    """
    first, second, bad = (make_candidate(1), make_candidate(2),
                          make_candidate(3))
    StubProvider.catalog = [first, bad, second]
    monkeypatch.setattr(
        visuals, "llm_screen",
        _keep_only({first.provider_id, second.provider_id}),
    )
    # The best keeper 404s. The next row in the ranking is the reject.
    sandbox.routes[first.download_url] = FakeResponse(404, body=b"gone")

    manifest = run_visuals(
        relevance_mode="llm", assets_per_beat=1, providers=["stub"],
    )
    # Scoped to the first beat: the stub serves one catalog to every beat, so a
    # later one runs out of keepers honestly and is meant to reach the reject.
    opening = manifest["beats"][0]["asset_indexes"]
    assert opening, "the opening beat sourced nothing at all"
    used = {manifest["assets"][i]["source_url"] for i in opening}
    assert bad.source_url not in used, (
        "reached past a surviving keeper for a clip the screen had rejected"
    )
    assert second.source_url in used


def test_a_reject_is_still_placed_when_it_is_the_only_thing_left(
    sandbox, monkeypatch
):
    """A blank beat stays the worst outcome, and the manifest says it happened.

    The rule is an ordering, not a ban: keepers, then the neutral shot, then -
    only once both have come back empty - a reject, with a warning naming the
    beat so it is never a silent substitution.
    """
    only = make_candidate(3)
    StubProvider.catalog = [only]
    # Nothing is kept, and there is no neutral query on a heuristic run with an
    # empty topic, so the reject is genuinely all there is.
    monkeypatch.setattr(visuals, "llm_screen", _keep_only(set()))

    manifest = run_visuals(
        relevance_mode="llm", assets_per_beat=1, providers=["stub"], topic="",
    )
    assert manifest["assets"], "left the reel with no footage at all"
    assert any("the relevance screen rejected" in w for w in manifest["warnings"])


def test_rejected_footage_is_kept_out_of_the_headroom_top_up(
    sandbox, monkeypatch, one_shot_beats
):
    """The top-up pulls spare shots from the leftovers.

    Rejected footage must not sit in there waiting to be collected as filler
    once each beat itself is full - that puts back on screen exactly what the
    screen just took off it, and without even a warning, because from the
    top-up's point of view it is only the next unused candidate.
    """
    catalog = [make_candidate(i) for i in range(14)]
    StubProvider.catalog = catalog
    keep = {c.provider_id for c in catalog[:3]}
    rejected_ids = set()
    monkeypatch.setattr(visuals, "llm_screen", _keep_only(keep, rejected_ids))

    manifest = run_visuals(
        relevance_mode="llm", assets_per_beat=1, headroom=4.0, providers=["stub"],
    )
    assert rejected_ids, "the screen rejected nothing, so this proves nothing"
    banned = {c.source_url for c in catalog if c.provider_id in rejected_ids}
    shipped = [a["source_url"] for a in manifest["assets"]]
    # The beats themselves can still reach a reject as a last resort, and say so.
    # The top-up may not: everything it adds is surplus, so a rejected clip
    # there is never the difference between a shot and a blank screen.
    topped_up = {a["source_url"] for a in manifest["assets"]
                 if a.get("role") == "spare"}
    assert topped_up, "the top-up added nothing, so this proves nothing"
    assert not (topped_up & banned), (
        f"the top-up placed rejected footage: {sorted(topped_up & banned)}"
    )
    assert shipped, "nothing was sourced at all"


def test_footage_below_the_shortlist_is_never_placed_silently(sandbox, monkeypatch):
    """The screen judges the shortlist. It says nothing about the rest.

    "Not rejected" was being read as "kept", so acquisition walked out of the
    judged rows and into the unjudged ones without a word. That is the shipwreck
    reel's beat 1: five power drills demoted, one keeper left for a two-clip
    beat, and a caterpillar and a man at Christmas dinner underneath it.
    """
    StubProvider.catalog = [make_candidate(i) for i in range(30)]
    keep = {c.provider_id for c in StubProvider.catalog[:2]}
    rejected_ids = set()
    monkeypatch.setattr(visuals, "llm_screen", _keep_only(keep, rejected_ids))

    manifest = run_visuals(
        relevance_mode="llm", assets_per_beat=2, providers=["stub"],
    )

    judged = keep | rejected_ids
    unjudged = [a for a in manifest["assets"]
                if a["provider_key"].split(":", 1)[1] not in judged]
    assert unjudged, "nothing came from below the shortlist, so this proves nothing"
    assert any("unscreened clip(s)" in w for w in manifest["warnings"]), (
        f"unjudged footage shipped without a word: {manifest['warnings']}"
    )

    notes = {b["index"]: b["notes"] for b in manifest["beats"]}
    for asset in unjudged:
        if asset.get("role") == "spare":
            continue           # the top-up reports once, at the end of the run
        assert any("never judged them" in note
                   for note in notes[asset["beat_index"]]), (
            f"beat {asset['beat_index']} placed unjudged footage with no note"
        )


def test_an_unscreened_run_does_not_hold_its_own_footage_back(sandbox):
    """The other half of the rule: no screen, no verdicts, no unjudged tail.

    "Unjudged" only means anything against a screen that ran. With the screen
    off every clip is unjudged, and demoting them all below a neutral shot
    would empty every beat in the reel rather than protect it.
    """
    StubProvider.catalog = [make_candidate(i) for i in range(30)]

    manifest = run_visuals(
        relevance_mode="off", assets_per_beat=1, providers=["stub"],
    )

    assert manifest["assets"], "left the reel with no footage at all"
    assert not any("unscreened clip(s)" in w for w in manifest["warnings"])
    for beat in manifest["beats"]:
        assert not any("never judged them" in note for note in beat["notes"])


def _keep_only_the_first(shortlists, beats, topic, subject_keys=None, establishing=False):
    """Keeps one clip of a beat's own footage, and all of the neutral shot's.

    The beat's own shortlist is cut to a single keeper so that every beat ends
    up short and reaches the fallback. The neutral's shortlist is kept whole,
    because the cap is about how many *different* beats take footage from it -
    if the screen left only one neutral clip in the run, the second beat would
    find it already spent and the cap would never be the thing that decided.
    """
    # `_screen_beats` screens every beat in one call; `_neutral_plan` screens
    # one beat on its own. That is the only thing here that tells the beat's
    # own footage from the neutral shot's, because in heuristic mode the topic
    # hint is both the neutral query and one of each beat's alt queries - so
    # the same clips arrive down both routes and no property of a candidate
    # can separate them.
    neutral_pass = len(shortlists) == 1
    verdicts: dict[int, dict[int, tuple[bool, str]]] = {}
    for index, candidates in shortlists.items():
        verdicts[index] = {
            position: (True, "") if neutral_pass or position == 0
            else (False, "shows the wrong thing")
            for position in range(len(candidates))
        }
    return verdicts, []


def _neutral_fills(manifest):
    return [b["index"] for b in manifest["beats"]
            if any(n.startswith("filled ") for n in b["notes"])]


def _neutral_serves_its_own_footage(monkeypatch):
    """Stub search where the neutral query alone returns unseen clips.

    The plain stub answers every query with one catalog, so the neutral shot
    can only ever return footage the beat has already ranked and is deduped
    away to nothing - the beat then falls through to unjudged rows and the
    cap is never what decided anything.
    """
    main = [make_candidate(i) for i in range(40)]
    # Worded distinctly, or the variety check reads the neutral clip as another
    # take of what the beat already holds ("clip 3" against "clip 103") and
    # skips it - which from the outside looks exactly like the cap firing.
    # Each one worded differently again: the variety check is run-level, so
    # eight clips all called "open water horizon" would let exactly one beat
    # take from the neutral and every later beat would be refused by variety
    # rather than by the cap.
    scenes = [
        "kelp forest drifting", "coral shelf sunlit", "sand ripples shallow",
        "open horizon calm", "tidal pool still", "cliff face spray",
        "lagoon glass flat", "reef edge blue",
    ]
    safe = [
        Candidate(
            provider="stub", provider_id=f"video{100 + i}", kind="video",
            width=1080, height=1920, duration=8.0,
            download_url=f"https://stub.test/video/{100 + i}.mp4",
            source_url=f"https://stub.test/page/{100 + i}",
            creator=f"neutral{i}", license="Stub License",
            title=f"{scene} horizon",
        )
        for i, scene in enumerate(scenes)
    ]

    def search(self, query, kind, limit):
        self.searches += 1
        rows = safe if query.strip().lower() == "ocean salty" else main
        return [c for c in rows if c.kind == kind][:limit]

    monkeypatch.setattr(StubProvider, "search", search)


def test_the_neutral_shot_stops_carrying_the_reel_once_it_hits_its_cap(
        sandbox, monkeypatch):
    """The beetle reel of 2026-09-14, which is why the cap exists.

    The neutral is one shot for the whole video, so each beat that reaches for
    it shows the same footage again. Four of that reel's eight beats came back
    as the same bark texture and the look pass graded it 1/3 "off_topic". Past
    the cap a beat takes footage the screen never judged instead: unjudged, but
    aimed at this beat rather than at the video in general.
    """
    _neutral_serves_its_own_footage(monkeypatch)
    monkeypatch.setattr(visuals, "llm_screen", _keep_only_the_first)
    monkeypatch.setattr(config, "VISUAL_MAX_NEUTRAL_BEATS", 1)

    manifest = run_visuals(
        relevance_mode="llm", assets_per_beat=2, providers=["stub"],
    )

    filled = _neutral_fills(manifest)
    assert len(filled) <= 1, f"the cap let the neutral take {len(filled)} beats"
    assert manifest["stats"]["beats_from_neutral"] == len(filled)
    # The beat that went without has to say so, or the cap is a silent
    # downgrade from a repeat to an unjudged clip.
    assert any(n.startswith("did not reach for the neutral shot")
               for b in manifest["beats"] for n in b["notes"])


def test_without_the_cap_the_neutral_shot_takes_every_beat_it_can(
        sandbox, monkeypatch):
    """The other half: -1 restores the behaviour the cap replaced.

    Pinned so the test above is known to be measuring the cap and not some
    other reason the neutral stopped being reached for.
    """
    _neutral_serves_its_own_footage(monkeypatch)
    monkeypatch.setattr(visuals, "llm_screen", _keep_only_the_first)
    monkeypatch.setattr(config, "VISUAL_MAX_NEUTRAL_BEATS", -1)

    manifest = run_visuals(
        relevance_mode="llm", assets_per_beat=2, providers=["stub"],
    )

    assert len(_neutral_fills(manifest)) > 1, (
        "nothing reached the neutral shot, so the capped run proves nothing"
    )


def test_the_cap_never_leaves_a_beat_holding_a_rejected_clip(
        sandbox, monkeypatch):
    """The cap may only ever swap a repeat for something unjudged.

    Below the unjudged rows sit clips the screen threw out, and one of those
    contradicts the narration - which is worse than seeing the safe shot
    again. With nothing unjudged to fall through to, a beat past the cap still
    gets the neutral.
    """
    # A catalog no deeper than the shortlist: everything is screened, so there
    # are no unjudged rows anywhere in the run.
    _neutral_serves_its_own_footage(monkeypatch)
    # A shortlist deeper than the pool: everything is screened, so the run has
    # no unjudged rows anywhere for a capped beat to fall through to.
    monkeypatch.setattr(config, "VISUAL_RELEVANCE_SHORTLIST", 200)
    monkeypatch.setattr(visuals, "llm_screen", _keep_only_the_first)
    monkeypatch.setattr(config, "VISUAL_MAX_NEUTRAL_BEATS", 0)

    manifest = run_visuals(
        relevance_mode="llm", assets_per_beat=2, providers=["stub"],
    )

    assert _neutral_fills(manifest), (
        "the cap fell through to rejected footage rather than the safe shot"
    )


def test_the_query_writer_is_asked_for_a_neutral_shot():
    """The fallback only exists if step 4 asks for it and can read it back."""
    prompt = visuals._QUERY_PROMPT_TEMPLATE
    assert '"neutral"' in prompt
    assert "no machinery" in prompt
    parsed = visuals._parse_query_json(
        '[{"index": 0, "query": "Andes terraces", "alt": "stone wall close up",'
        ' "neutral": "Andes mountain landscape", "image_prompt": "x"}]', 1,
    )
    assert parsed[0]["neutral"] == "Andes mountain landscape"


def test_a_query_without_a_neutral_still_parses():
    """An older or truncated response must cost the fallback, never the run."""
    parsed = visuals._parse_query_json(
        '[{"index": 0, "query": "Andes terraces", "alt": "stone wall"}]', 1,
    )
    assert parsed[0]["neutral"] == ""


def test_the_query_writer_is_told_that_bare_ancient_does_not_search(monkeypatch):
    """The queries that produced the worst footage were era-stripped.

    "ancient construction workers" returned a cement mixer and "long ancient
    highway" a motorway with painted lane markings, because a stock library
    matches the words it indexes and "ancient" is not one of them.
    """
    prompt = visuals._QUERY_PROMPT_TEMPLATE
    assert "proper" in prompt and "noun" in prompt
    assert "cement mixer" in prompt
    # The alt must not be a licence to drop the subject, which is how a beat
    # about the Inca reached an office worker.
    assert "still in the right world" in prompt


def test_lexical_mode_does_not_research_when_it_cannot_tell(monkeypatch):
    """All-zero is unjudgeable to the lexical check, not evidence of a bad beat.

    Deliberate, and the reason the re-search belongs to the model: "brown bat
    sleeping" scores every correct flying-fox clip at zero too, so treating a
    zero sweep as grounds to go back out would re-search good beats and then
    promote the fallback above footage that was right all along.
    """
    beat = Beat(index=0, start=0.0, end=6.0, query="brown bat sleeping",
                text="Brown bats sleep for nineteen hours.",
                alt_queries=("bat colony",))
    searched: list[str] = []

    class CountingProvider(visuals.Provider):
        name = "stub"
        kinds = ("video",)
        key_env = ()
        attribution_required = False

        def search(self, query, kind, limit):
            searched.append(query)
            return []

    plans = {0: [(9.0, _described(30, "grey-headed-flying-foxes-are-amazing"), "q")]}
    visuals._screen_beats(
        plans, [beat], "sleep", [CountingProvider(session=None)],
        _settings(relevance_mode="lexical", relevance_floor=0.15,
                  relevance_shortlist=6, kinds=["video"]),
        visuals.UsageLedger(), "run", visuals._RunState(), {}, [],
    )
    assert searched == []                    # no quota spent
    assert plans[0][0][1].provider_id == "v30"   # and nothing demoted


def test_the_manifest_records_what_each_clip_says_it_shows(sandbox):
    StubProvider.catalog = [make_candidate(i) for i in range(1, 5)]
    manifest = run_visuals()
    for asset in manifest["assets"]:
        assert "describes" in asset
        assert 0.0 <= asset["relevance"] <= 1.0


def test_relevance_off_screens_nothing_and_costs_no_call(sandbox):
    StubProvider.catalog = [make_candidate(i) for i in range(1, 5)]
    manifest = run_visuals(relevance_mode="off")
    assert manifest["assets"]
    assert not any("screen" in w.lower() for w in manifest["warnings"])


# ---------------------------------------------------------------------------
# Variety - two files, one shot
#
# Both cases below are from the sleep reel, which spent four of its last six
# shots on two men in two beds. One pair was caught by wording, the other only
# by the creator: stock libraries sell a whole shoot as separate clips.
# ---------------------------------------------------------------------------

def _variety_settings(**overrides):
    base = {"similar_overlap": config.VISUAL_SIMILAR_OVERLAP,
            "max_per_creator": config.VISUAL_MAX_PER_CREATOR}
    base.update(overrides)
    return base


def _by(creator: str, index: int, slug: str) -> Candidate:
    return Candidate(
        provider="stub", provider_id=f"v{index}", kind="video",
        width=1080, height=1920, duration=8.0,
        download_url=f"https://stub.test/video/{index}.mp4",
        source_url=f"https://www.pexels.com/video/{slug}-{3000 + index}/",
        creator=creator,
    )


def test_a_second_clip_of_the_same_shot_is_recognised_by_its_wording():
    state = visuals._RunState()
    state.remember(_by("Katrin", 1, "man-feeling-sleepy"))
    repeat = _by("Nicola", 2, "video-of-a-man-sleeping")

    assert state.too_similar(repeat, _variety_settings())


def test_two_different_shots_of_one_subject_are_not_the_same_shot():
    # Both gorillas, and both worth having: this is the case the threshold
    # must leave alone, or a beat loses its only usable pair.
    state = visuals._RunState()
    state.remember(_by("Connor", 3, "gorilla-in-autumn-forest-eating-leaves"))
    other = _by("David", 4, "majestic-gorilla-walking-in-lush-habitat")

    assert not state.too_similar(other, _variety_settings())


def test_one_creator_does_not_supply_two_clips_of_a_run():
    # "relaxing rest tired indoors" and "video of a man sleeping" share no
    # word at all, and are the same man on the same bed by the same person.
    state = visuals._RunState()
    state.remember(_by("Nicola Barts", 5, "relaxing-rest-tired-indoors"))
    same_shoot = _by("Nicola Barts", 6, "video-of-a-man-sleeping")

    assert "per creator" in state.too_similar(same_shoot, _variety_settings())


def test_the_creator_cap_can_be_lifted():
    state = visuals._RunState()
    state.remember(_by("Nicola Barts", 7, "relaxing-rest-tired-indoors"))
    same_shoot = _by("Nicola Barts", 8, "video-of-a-man-sleeping")

    assert not state.too_similar(
        same_shoot, _variety_settings(max_per_creator=0, similar_overlap=0.45)
    )


def test_a_clip_with_no_description_is_never_called_a_repeat():
    state = visuals._RunState()
    state.remember(_by("Someone", 9, "a-man-sleeping"))
    blank = Candidate(provider="stub", provider_id="z", kind="video",
                      width=1080, height=1920, download_url="https://stub.test/z.mp4")

    assert not state.too_similar(blank, _variety_settings())


def test_variety_never_leaves_a_beat_short(sandbox):
    """The preference must yield to the thing it exists to prevent.

    Every clip here is by one creator, so the strict pass can take exactly one
    of them. The beat still has to come away with two, because one clip over a
    whole beat is the held shot the rule is trying to avoid.
    """
    StubProvider.catalog = [
        Candidate(provider="stub", provider_id=f"v{i}", kind="video",
                  width=1080, height=1920, duration=9.0,
                  download_url=f"https://stub.test/video/{i}.mp4",
                  source_url=f"https://www.pexels.com/video/one-shoot-{i}/",
                  creator="The Only Creator")
        for i in range(6)
    ]
    manifest = run_visuals(assets_per_beat=2)

    for record in manifest["beats"]:
        if record["status"] == "ok":
            assert len(record["asset_indexes"]) >= 2


def test_variety_spreads_a_run_across_creators_when_it_can(sandbox):
    StubProvider.catalog = [
        Candidate(provider="stub", provider_id=f"v{i}", kind="video",
                  width=1080, height=1920, duration=9.0,
                  download_url=f"https://stub.test/video/{i}.mp4",
                  source_url=f"https://www.pexels.com/video/subject-{i}-alone/",
                  creator=f"creator{i}")
        for i in range(12)
    ]
    manifest = run_visuals(assets_per_beat=2)
    creators = [a["creator"] for a in manifest["assets"]]

    assert len(set(creators)) == len(creators), creators


# ---------------------------------------------------------------------------
# Query construction
# ---------------------------------------------------------------------------

def test_heuristic_queries_drop_stopwords_and_keep_subjects():
    beats = [Beat(0, "The water evaporates and the salt stays behind.", 0.0, 4.0)]
    visuals.heuristic_queries(beats, "why the ocean is salty")
    terms = beats[0].query.split()
    assert terms and not (set(terms) & visuals._STOPWORDS)
    assert "evaporates" in terms or "water" in terms
    assert beats[0].query_source == "heuristic"
    assert visuals.IMAGE_PROMPT_STYLE in beats[0].image_prompt


def test_heuristic_queries_offer_a_broader_fallback():
    beats = [Beat(0, "Rain dissolves granite into sodium and chloride.", 0.0, 4.0)]
    visuals.heuristic_queries(beats, "why the ocean is salty")
    assert beats[0].alt_queries
    assert beats[0].query not in beats[0].alt_queries


def test_beat_with_only_stopwords_still_gets_a_query():
    beats = [Beat(0, "And so it was that they did.", 0.0, 4.0)]
    visuals.heuristic_queries(beats, "the sea")
    assert beats[0].query.strip()


def test_the_query_prompt_asks_for_self_contained_queries():
    """A live run wrote 'salt crystals close up' for a video about the ocean
    and Pexels returned a kitchen worktop. The query reaches the library with
    no knowledge of the video, so the prompt has to say so."""
    prompt = visuals._QUERY_PROMPT_TEMPLATE.format(
        topic="why the ocean is salty", beats='0. (4.0s) "The ocean is salty."'
    )
    assert "never heard of this video" in prompt
    assert "sea salt crystals" in prompt, "the worked example is the useful part"


def test_parse_query_json_handles_fences_and_junk_entries():
    raw = """```json
    [{"index": 0, "query": "ocean waves", "alt": "sea", "image_prompt": "a wave"},
     "not an object",
     {"index": 99, "query": "out of range"},
     {"index": 1, "query": "", "alt": "empty query is dropped"},
     {"index": 2, "query": "rain on rock"}]
    ```"""
    parsed = visuals._parse_query_json(raw, expected=3)
    assert set(parsed) == {0, 2}
    assert parsed[0]["query"] == "ocean waves"
    assert parsed[2]["alt"] == ""


def test_parse_query_json_rejects_a_non_array():
    with pytest.raises(ValueError):
        visuals._parse_query_json('{"query": "nope"}', expected=1)


def test_llm_mode_falls_back_to_keywords_without_a_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    beats = visuals.segment_script(SCRIPT, 40.0)
    warnings = visuals.build_queries(beats, "the ocean", "llm")

    assert warnings and "GEMINI_API_KEY" in warnings[0]
    assert all(b.query and b.query_source == "heuristic" for b in beats)


def test_unknown_query_mode_warns_but_still_produces_queries():
    beats = visuals.segment_script(SCRIPT, 40.0)
    warnings = visuals.build_queries(beats, "the ocean", "telepathy")
    assert warnings and "unknown" in warnings[0]
    assert all(b.query for b in beats)


# ---------------------------------------------------------------------------
# Provider response parsing
# ---------------------------------------------------------------------------

def test_pexels_videos_parse_and_pick_the_cheapest_passing_rendition():
    payload = {"videos": [{
        "id": 7, "duration": 12, "url": "https://pexels.com/video/7",
        "image": "prev.jpg", "alt": "a wave", "user": {"name": "Ana", "url": "u"},
        "video_files": [
            {"link": "sd.mp4", "width": 426, "height": 758},       # under any floor
            {"link": "hd.mp4", "width": 1080, "height": 1920},     # cheapest pass
            {"link": "uhd.mp4", "width": 2160, "height": 3840},    # overkill
        ],
    }]}
    provider = visuals.PexelsProvider(FakeSession())
    [candidate] = provider._videos(payload)

    assert candidate.download_url == "hd.mp4"
    assert (candidate.width, candidate.height) == (1080, 1920)
    assert candidate.duration == 12.0
    assert candidate.creator == "Ana"
    assert candidate.key == "pexels:7"
    assert "Pexels License" in candidate.attribution


def test_pexels_video_without_usable_files_is_dropped():
    provider = visuals.PexelsProvider(FakeSession())
    assert provider._videos({"videos": [{"id": 1, "video_files": []}]}) == []


def test_pixabay_photos_report_the_size_actually_served():
    """largeImageURL caps the longest edge at 1280 whatever the original was."""
    payload = {"hits": [{
        "id": 3, "pageURL": "p", "largeImageURL": "big.jpg",
        "imageWidth": 4000, "imageHeight": 6000, "user": "Bo", "tags": "sea",
    }]}
    provider = visuals.PixabayProvider(FakeSession())
    [candidate] = provider._photos(payload)

    assert candidate.height == 1280 and candidate.width == 853
    # Never clears the full 1080x1920 frame, so a Pixabay photo is always an
    # upscale. It does clear a lowered floor, which is how it can be sourced at
    # all - see VISUAL_MIN_WIDTH/HEIGHT.
    assert not visuals.meets_resolution_floor(
        candidate.width, candidate.height, **FRAME_FLOOR
    )
    assert visuals.meets_resolution_floor(
        candidate.width, candidate.height, min_width=608, min_height=1080
    )


def test_pixabay_videos_take_the_largest_stream():
    payload = {"hits": [{
        "id": 9, "duration": 20, "pageURL": "p", "user": "Cy", "tags": "rain",
        "videos": {
            "large": {"url": "l.mp4", "width": 1080, "height": 1920},
            "medium": {"url": "m.mp4", "width": 720, "height": 1280},
            "tiny": {"url": "t.mp4", "width": 0, "height": 0},
        },
    }]}
    provider = visuals.PixabayProvider(FakeSession())
    [candidate] = provider._videos(payload)
    assert candidate.download_url == "l.mp4" and candidate.height == 1920


def test_openverse_carries_the_licence_into_the_attribution():
    session = FakeSession(routes={"openverse": FakeResponse(200, payload={"results": [{
        "id": "abc", "url": "img.jpg", "width": 2000, "height": 3000,
        "creator": "Dee", "license": "by", "license_version": "4.0",
        "title": "a shore", "foreign_landing_url": "https://src/page",
    }]})})
    [candidate] = visuals.OpenverseProvider(session).search("shore", "image", 5)

    assert candidate.license == "CC BY 4.0"
    assert "Dee" in candidate.attribution and "https://src/page" in candidate.attribution
    assert visuals.OpenverseProvider.attribution_required is True


def test_openverse_has_no_video_endpoint():
    assert visuals.OpenverseProvider(FakeSession()).search("x", "video", 5) == []


# ---------------------------------------------------------------------------
# YouTube (Creative Commons only)
# ---------------------------------------------------------------------------

def _youtube_session(items, search_ids=None):
    """A session answering both halves of the youtube provider's two calls."""
    ids = search_ids if search_ids is not None else [i["id"] for i in items]
    return FakeSession(routes={
        "v3/search": FakeResponse(200, payload={
            "items": [{"id": {"videoId": vid}} for vid in ids]
        }),
        "v3/videos": FakeResponse(200, payload={"items": items}),
    })


def _yt_item(video_id="v1", *, license="creativeCommon", duration="PT2M10S",
             channel="Deep Sea Trust"):
    return {
        "id": video_id,
        "status": {"license": license},
        "contentDetails": {"duration": duration, "definition": "hd"},
        "snippet": {
            "title": f"clip {video_id}", "channelTitle": channel,
            "channelId": "UC123", "thumbnails": {"high": {"url": "thumb.jpg"}},
        },
    }


def test_youtube_dimensions_come_from_the_probe_not_the_api(monkeypatch):
    """The API only says "hd", which does not separate 1080p from 4K - and that
    is exactly the distinction the 9:16 floor turns on."""
    monkeypatch.setattr(visuals, "_ytdl_probe", lambda url: (3840, 2160))
    session = _youtube_session([_yt_item()])
    [candidate] = visuals.YouTubeCCProvider(session).search("deep sea", "video", 5)

    assert (candidate.width, candidate.height) == (3840, 2160)
    assert visuals.meets_resolution_floor(
        candidate.width, candidate.height, min_width=1080, min_height=1920
    )


def _video_settings(**over):
    base = {"kinds": ("video", "image"), "allow_landscape": True,
            "min_width": 1080, "min_height": 1920, "provider_floors": {}}
    return {**base, **over}


def test_a_1080p_youtube_upload_cannot_clear_the_global_floor(monkeypatch):
    """The trap the per-provider floor exists to solve: "HD" is not enough for
    a 16:9 source, because the floor is applied to what survives the crop."""
    monkeypatch.setattr(visuals, "_ytdl_probe", lambda url: (1920, 1080))
    session = _youtube_session([_yt_item()])
    [candidate] = visuals.YouTubeCCProvider(session).search("deep sea", "video", 5)

    reason = visuals.reject_reason(candidate, _video_settings())
    assert "608x1080" in reason and "1080x1920 floor" in reason


def test_a_1080p_youtube_upload_passes_under_youtubes_own_floor(monkeypatch):
    monkeypatch.setattr(visuals, "_ytdl_probe", lambda url: (1920, 1080))
    session = _youtube_session([_yt_item()])
    [candidate] = visuals.YouTubeCCProvider(session).search("deep sea", "video", 5)

    settings = _video_settings(provider_floors={"youtube": (608, 1080)})
    assert visuals.reject_reason(candidate, settings) == ""


def test_lowering_youtubes_floor_does_not_lower_anyone_elses():
    """The whole point of making this per-provider: weak stock stays out."""
    settings = _video_settings(provider_floors={"youtube": (608, 1080)})
    stock = make_candidate(1, width=1920, height=1080, provider="pexels")

    assert "under the 1080x1920 floor" in visuals.reject_reason(stock, settings)


def test_floor_for_falls_back_to_the_global_pair():
    settings = _video_settings(provider_floors={"youtube": (608, 1080)})
    assert visuals.floor_for("youtube", settings) == (608, 1080)
    assert visuals.floor_for("pexels", settings) == (1080, 1920)
    assert visuals.floor_for("pexels", _video_settings()) == (1080, 1920)


@pytest.mark.parametrize("raw,expected", [
    ("youtube:608x1080", {"youtube": (608, 1080)}),
    ("youtube:608x1080,nasa:900x1600", {"youtube": (608, 1080), "nasa": (900, 1600)}),
    ("  YouTube : 608X1080 ", {"youtube": (608, 1080)}),
    ("", {}),
    (None, {}),
    ("youtube", {}),                       # no size at all
    ("youtube:wide", {}),                  # unparseable size
    ("youtube:608x1080,broken", {"youtube": (608, 1080)}),   # one bad pair only
])
def test_provider_floors_parse_forgivingly(raw, expected):
    """A typo in one pair costs that provider its override, nothing more."""
    assert config._parse_provider_floors(raw) == expected


def test_changing_a_provider_floor_invalidates_the_cached_manifest():
    """Otherwise a run whose floor had moved would be served the old footage."""
    base = {
        "providers": ["youtube"], "assets_per_beat": 2, "headroom": 1.6,
        "min_width": 1080, "min_height": 1920, "kinds": ("video",),
        "allow_landscape": True, "query_mode": "llm", "image_duration": 4.0,
        "relevance_mode": "llm", "relevance_floor": 0.2, "relevance_shortlist": 6,
        "provider_floors": {}, "max_neutral_beats": 2,
    }
    moved = {**base, "provider_floors": {"youtube": (608, 1080)}}

    assert visuals._settings_fingerprint(base) != visuals._settings_fingerprint(moved)


def test_youtube_skips_a_video_the_probe_cannot_play(monkeypatch):
    """Private, geo-blocked and members-only all look the same from here."""
    monkeypatch.setattr(visuals, "_ytdl_probe", lambda url: None)
    session = _youtube_session([_yt_item()])
    assert visuals.YouTubeCCProvider(session).search("deep sea", "video", 5) == []


def test_youtube_claims_no_licence_while_its_guards_are_off():
    """The guards are commented out, so there is no grant to assert.

    This pinned `attribution_required is True` and a `CC BY 3.0` stamp until
    2026-09-17, which is the combination that credited an All Rights Reserved
    upload as Creative Commons. The guards stayed off and the claim went; see
    the `YouTubeCCProvider` docstring for the trade-off that leaves standing.
    """
    provider = visuals.YouTubeCCProvider
    assert provider.attribution_required is False
    assert provider.LICENSE == ""
    assert provider.key_env == ("YOUTUBE_API_KEY",)


def test_youtube_attribution_names_the_channel_and_asserts_no_licence(monkeypatch):
    """The channel is still named - that part was never the problem.

    Crediting the uploader is right whatever the licence turns out to be. What
    must not appear is a licence this provider never read off the upload, so
    the parenthetical falls back to "see source" rather than naming one.
    """
    monkeypatch.setattr(visuals, "_ytdl_probe", lambda url: (3840, 2160))
    session = _youtube_session([_yt_item(channel="Deep Sea Trust")])
    [candidate] = visuals.YouTubeCCProvider(session).search("deep sea", "video", 5)

    assert candidate.creator == "Deep Sea Trust"
    assert candidate.license == ""
    assert "CC BY" not in candidate.attribution
    assert "(see source)" in candidate.attribution
    assert "youtube.com/watch?v=v1" in candidate.attribution


def test_youtube_caps_how_many_hits_it_probes(monkeypatch):
    """Each probe is a round trip to the watch page, so the search is not
    allowed to spend one on every result."""
    probed = []

    def probe(url):
        probed.append(url)
        return (3840, 2160)

    monkeypatch.setattr(visuals, "_ytdl_probe", probe)
    session = _youtube_session([_yt_item(f"v{i}") for i in range(20)])
    visuals.YouTubeCCProvider(session).search("deep sea", "video", 20)

    assert len(probed) == visuals.YOUTUBE_PROBE_LIMIT


def test_youtube_has_no_image_endpoint():
    assert visuals.YouTubeCCProvider(FakeSession()).search("x", "image", 5) == []


def test_youtube_skips_a_live_stream_reporting_no_duration(monkeypatch):
    monkeypatch.setattr(visuals, "_ytdl_probe", lambda url: (3840, 2160))
    session = _youtube_session([_yt_item(duration="P0D")])
    assert visuals.YouTubeCCProvider(session).search("x", "video", 5) == []


@pytest.mark.parametrize("text,seconds", [
    ("PT4M13S", 253.0),
    ("PT1H2M3S", 3723.0),
    ("PT45S", 45.0),
    ("PT2M", 120.0),
    ("P0D", 0.0),
    ("", 0.0),
    ("garbage", 0.0),
])
def test_iso8601_durations_parse(text, seconds):
    assert visuals._parse_iso8601_duration(text) == seconds


# ---------------------------------------------------------------------------
# NASA
# ---------------------------------------------------------------------------

def _nasa_session(files, *, nasa_id="PIA12345"):
    return FakeSession(routes={
        "images-api.nasa.gov/search": FakeResponse(200, payload={"collection": {
            "items": [{
                "href": "https://images-assets.nasa.gov/collection.json",
                "data": [{"nasa_id": nasa_id, "title": "Europa", "center": "JPL"}],
                "links": [{"href": "thumb.jpg"}],
            }],
        }}),
        "collection.json": FakeResponse(200, payload=files),
    })


def test_nasa_prefers_the_original_rendition():
    session = _nasa_session([
        "https://x/PIA12345~thumb.jpg",
        "https://x/PIA12345~medium.jpg",
        "https://x/PIA12345~orig.jpg",
        "https://x/PIA12345~large.jpg",
    ])
    [candidate] = visuals.NasaProvider(session).search("europa", "image", 5)

    assert candidate.download_url.endswith("~orig.jpg")
    assert candidate.provider_id == "PIA12345"
    assert candidate.license == "Public domain (NASA)"


def test_nasa_falls_back_down_the_rendition_ladder():
    session = _nasa_session(["https://x/PIA12345~medium.jpg"])
    [candidate] = visuals.NasaProvider(session).search("europa", "image", 5)
    assert candidate.download_url.endswith("~medium.jpg")


def test_nasa_skips_an_item_whose_manifest_holds_no_image():
    session = _nasa_session(["https://x/PIA12345~orig.tif", "https://x/meta.json"])
    assert visuals.NasaProvider(session).search("europa", "image", 5) == []


def test_nasa_survives_a_dead_manifest():
    """One unreachable manifest is a skipped item, not a failed search."""
    session = FakeSession(routes={
        "images-api.nasa.gov/search": FakeResponse(200, payload={"collection": {
            "items": [{
                "href": "https://images-assets.nasa.gov/collection.json",
                "data": [{"nasa_id": "PIA1", "title": "t", "center": "JPL"}],
            }],
        }}),
        "collection.json": FakeResponse(500, body=b"nope"),
    })
    assert visuals.NasaProvider(session).search("europa", "image", 5) == []


def test_nasa_claims_no_more_than_the_floor_it_must_clear():
    """The nominal size only buys a ticket past reject_reason, which drops
    anything reporting no dimensions. probe_image_size settles it for real."""
    assert visuals.NasaProvider.nominal_size() == (
        config.VISUAL_MIN_WIDTH, config.VISUAL_MIN_HEIGHT
    )


def test_nasa_needs_no_key_and_owes_no_credit():
    assert visuals.NasaProvider.is_configured()
    assert visuals.NasaProvider.attribution_required is False


def test_nasa_has_no_video_endpoint():
    assert visuals.NasaProvider(FakeSession()).search("x", "video", 5) == []


# ---------------------------------------------------------------------------
# The ytdl:// download path
# ---------------------------------------------------------------------------

def test_a_ytdl_url_never_reaches_the_http_session(monkeypatch, tmp_path):
    """youtube publishes no direct media link, so the scheme is a hand-off."""
    handled = {}

    def fake(url, destination, *, provider):
        handled["url"] = url
        return {"bytes": 1, "sha256": "d", "content_type": "video/mp4"}

    monkeypatch.setattr(visuals, "download_via_ytdl", fake)
    session = FakeSession()
    visuals.download(session, "ytdl://abc123", tmp_path / "f.mp4", provider="youtube")

    assert handled["url"] == "ytdl://abc123"
    assert session.calls == []


def test_a_ytdl_url_is_cached_under_mp4():
    """path_for reads the extension off the URL, and ytdl:// carries none."""
    assert visuals._extension_for("ytdl://abc123") == ".mp4"


def test_ytdl_download_without_the_library_says_what_to_install(monkeypatch):
    monkeypatch.setitem(sys.modules, "yt_dlp", None)
    with pytest.raises(visuals.ProviderError, match="pip install yt-dlp"):
        visuals._import_ytdl("youtube")


def test_both_new_providers_are_registered():
    assert visuals._PROVIDERS["youtube"] is visuals.YouTubeCCProvider
    assert visuals._PROVIDERS["nasa"] is visuals.NasaProvider
    for name in ("youtube", "nasa"):
        assert name in config.VISUAL_PROVIDER_KEYS


def test_the_new_providers_are_opt_in():
    """Enabling youtube spends the topic scout's quota and needs yt-dlp, so it
    does not belong in a default run. Nor does nasa, which has nothing to say
    outside space and earth science.

    This asserts on the *default*, not on `config.VISUAL_PROVIDERS`. Reading
    the resolved value made the test fail the moment a machine opted in, which
    is the one thing an opt-in is supposed to allow: it was measuring this
    checkout's .env rather than the code's shipped default, and an .env is not
    what ships. `config.py` loads .env on import, so reloading the module does
    not get back to the default either - only the named constant does.
    """
    default = config.get_env_list(
        "_UNSET_PROVIDERS_", config.VISUAL_PROVIDERS_DEFAULT
    )
    assert "youtube" not in default
    assert "nasa" not in default
    assert default == ("pexels", "pixabay", "openverse")


def test_provider_readiness_reflects_the_configured_keys(monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", "k")
    assert visuals.PexelsProvider.is_configured()
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.delenv("STOCK_API_KEY", raising=False)
    assert not visuals.PexelsProvider.is_configured()
    assert visuals.OpenverseProvider.is_configured()   # keyless, always ready


def test_legacy_stock_api_key_still_works_for_pexels(monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.setenv("STOCK_API_KEY", "legacy")
    assert visuals.PexelsProvider.api_key() == "legacy"


# ---------------------------------------------------------------------------
# HTTP error mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status,expected", [
    (429, visuals.ProviderRateLimitError),
    (401, visuals.ProviderAuthError),
    (403, visuals.ProviderAuthError),
    (500, visuals.ProviderError),
])
def test_search_errors_map_to_the_right_class(status, expected):
    session = FakeSession(default=FakeResponse(status, body=b"upstream said no"))
    with pytest.raises(expected) as exc:
        visuals._get_json(session, "https://x/", provider="stub")
    assert "upstream said no" in str(exc.value)


def test_rate_limit_and_auth_failures_are_fatal_but_a_500_is_not():
    session = FakeSession(default=FakeResponse(429))
    with pytest.raises(visuals.ProviderError) as rate_limited:
        visuals._get_json(session, "https://x/", provider="stub")
    assert rate_limited.value.fatal

    session = FakeSession(default=FakeResponse(503))
    with pytest.raises(visuals.ProviderError) as transient:
        visuals._get_json(session, "https://x/", provider="stub")
    assert not transient.value.fatal


def test_non_json_body_is_a_provider_error():
    session = FakeSession(default=FakeResponse(200, body=b"<html>nope</html>"))
    with pytest.raises(visuals.ProviderError, match="non-JSON"):
        visuals._get_json(session, "https://x/", provider="stub")


def test_network_failure_is_a_provider_error(monkeypatch):
    import requests

    class Boom(FakeSession):
        def get(self, *a, **k):
            raise requests.exceptions.ConnectionError("dns failure")

    with pytest.raises(visuals.ProviderError, match="could not reach"):
        visuals._get_json(Boom(), "https://x/", provider="stub")


# ---------------------------------------------------------------------------
# Downloading
# ---------------------------------------------------------------------------

def test_download_writes_the_file_and_hashes_it(tmp_path):
    body = b"some-media-bytes"
    session = FakeSession(default=FakeResponse(
        200, body=body, headers={"Content-Type": "video/mp4"}
    ))
    destination = tmp_path / "nested" / "clip.mp4"
    result = visuals.download(session, "https://x/clip.mp4", destination, provider="stub")

    assert destination.read_bytes() == body
    assert result["bytes"] == len(body)
    assert result["content_type"] == "video/mp4"
    assert len(result["sha256"]) == 64


def test_dead_url_raises_without_leaving_a_file(tmp_path):
    session = FakeSession(default=FakeResponse(404, body=b"gone"))
    destination = tmp_path / "clip.mp4"
    with pytest.raises(visuals.ProviderError, match="404"):
        visuals.download(session, "https://x/clip.mp4", destination, provider="stub")
    assert not destination.exists()
    assert list(tmp_path.iterdir()) == []


def test_html_served_as_media_is_treated_as_a_dead_link(tmp_path):
    session = FakeSession(default=FakeResponse(
        200, body=b"<html>removed</html>", headers={"Content-Type": "text/html"}
    ))
    with pytest.raises(visuals.ProviderError, match="instead of media"):
        visuals.download(session, "https://x/a.mp4", tmp_path / "a.mp4", provider="stub")


def test_empty_body_is_rejected(tmp_path):
    session = FakeSession(default=FakeResponse(
        200, body=b"", headers={"Content-Type": "video/mp4"}
    ))
    with pytest.raises(visuals.ProviderError, match="empty body"):
        visuals.download(session, "https://x/a.mp4", tmp_path / "a.mp4", provider="stub")
    assert not (tmp_path / "a.mp4.part").exists()


class TruncatedResponse(FakeResponse):
    """Headers arrive, then the body breaks off part-way through.

    What a real ChunkedEncodingError looks like from `download`'s side: the
    status and Content-Type are fine, some bytes land, and then the connection
    dies. This is the shape that lost three beats on the deep-ocean run.
    """

    #: Bytes per yielded chunk. Fixed rather than taking `chunk_size`, so the
    #: break lands mid-body whatever DOWNLOAD_CHUNK_SIZE is set to - at its
    #: real 64KB the whole test body would arrive in one piece and nothing
    #: would break at all.
    piece = 64

    def __init__(self, body=b"a-real-clip-" * 64, after=2, headers=None):
        super().__init__(200, body=body,
                         headers=headers or {"Content-Type": "video/mp4"})
        self.after = after

    def iter_content(self, chunk_size=8192):
        import requests

        for index, start in enumerate(range(0, len(self.content), self.piece)):
            if index >= self.after:
                raise requests.exceptions.ChunkedEncodingError(
                    "Connection broken: IncompleteRead(128 bytes read, "
                    "19726597 more expected)"
                )
            yield self.content[start:start + self.piece]


def test_a_broken_stream_is_a_provider_error_not_a_raw_requests_error(tmp_path):
    """The whole point: `_acquire` only steps over `ProviderError`.

    A bare requests exception escaped past it, past the beat loop's own
    handlers, and cost the beat its footage plus every fallback below it.
    """
    session = FakeSession(default=TruncatedResponse())
    destination = tmp_path / "clip.mp4"

    with pytest.raises(visuals.ProviderTransientError, match="broke off"):
        visuals.download(session, "https://x/clip.mp4", destination,
                         provider="stub")

    assert isinstance(visuals.ProviderTransientError("x"), visuals.ProviderError)
    assert not destination.exists()
    assert not (tmp_path / "clip.mp4.part").exists(), "left a partial file behind"


def test_a_broken_stream_does_not_bench_the_provider(tmp_path):
    """A dropped connection says nothing about the provider's health."""
    session = FakeSession(default=TruncatedResponse())
    with pytest.raises(visuals.ProviderTransientError) as caught:
        visuals.download(session, "https://x/a.mp4", tmp_path / "a.mp4",
                         provider="stub")
    assert caught.value.fatal is False


def test_a_broken_stream_buys_one_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(visuals.time, "sleep", lambda _: None)
    attempts = []

    def flaky(url):
        attempts.append(url)
        if len(attempts) == 1:
            return TruncatedResponse()
        return unique_media(url)

    session = FakeSession(default=flaky)
    candidate = make_candidate(0)
    result = visuals._download_with_backoff(session, candidate, tmp_path / "a.mp4")

    assert len(attempts) == 2, "a truncated body should be tried once more"
    assert result["bytes"] > 0


def test_a_dead_url_is_not_retried(tmp_path, monkeypatch):
    """Only transient failures buy the retry - a 404 serves the same bytes."""
    monkeypatch.setattr(visuals.time, "sleep", lambda _: None)
    attempts = []

    def gone(url):
        attempts.append(url)
        return FakeResponse(404, body=b"gone")

    session = FakeSession(default=gone)
    with pytest.raises(visuals.ProviderError, match="404"):
        visuals._download_with_backoff(session, make_candidate(0),
                                       tmp_path / "a.mp4")
    assert len(attempts) == 1


def test_a_broken_stream_costs_one_asset_not_the_beat(sandbox, monkeypatch):
    """The regression this whole fix exists for.

    Before it, one truncated body took the beat down and skipped the
    neutral-shot and reject-recovery fallbacks with it.
    """
    monkeypatch.setattr(visuals.time, "sleep", lambda _: None)
    broken = "/0.mp4"

    def route(url):
        if url.endswith(broken):
            return TruncatedResponse()
        return unique_media(url)

    monkeypatch.setattr(visuals, "_new_session", lambda: FakeSession(default=route))
    manifest = run_visuals()

    beat = next(b for b in manifest["beats"] if b["index"] == 0)
    assert beat["status"] == "ok", "a broken stream took the beat down with it"
    assert len(beat["asset_indexes"]) == beat["shots"], (
        "the beat fell short of its shots"
    )
    assert not any(a["provider_id"] == "video0" for a in manifest["assets"])
    assert any("broke off" in w for w in manifest["warnings"])
    # And it costs nothing anywhere else either. This used to read
    # `beats_failed == 1`, which was never about the broken stream: the stub
    # serves one page to every beat, and at a page of 15 the eighth beat ran
    # out of catalogue whatever happened here.
    assert manifest["stats"]["beats_failed"] == 0


def test_oversized_file_is_refused_before_download(tmp_path):
    session = FakeSession(default=FakeResponse(200, body=b"x", headers={
        "Content-Type": "video/mp4",
        "Content-Length": str(visuals.MAX_DOWNLOAD_BYTES + 1),
    }))
    with pytest.raises(visuals.ProviderError, match="over the"):
        visuals.download(session, "https://x/a.mp4", tmp_path / "a.mp4", provider="stub")


def test_probe_image_size_measures_the_real_file(tmp_path):
    path = tmp_path / "shot.jpg"
    path.write_bytes(jpeg_bytes(1080, 1920))
    assert visuals.probe_image_size(path) == (1080, 1920)
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"not an image")
    assert visuals.probe_image_size(bad) is None


# ---------------------------------------------------------------------------
# Cache and usage ledger
# ---------------------------------------------------------------------------

def test_cache_round_trips_and_forgets_deleted_files(tmp_path):
    cache = visuals.AssetCache(tmp_path)
    media = tmp_path / "stub" / "a.mp4"
    media.parent.mkdir(parents=True)
    media.write_bytes(b"bytes")

    cache.put("stub:1", media, {"sha256": "deadbeef", "width": 1080, "height": 1920})
    cache.save()

    reloaded = visuals.AssetCache(tmp_path)
    assert reloaded.get("stub:1")["path"] == media
    assert reloaded.key_for_hash("deadbeef") == "stub:1"

    media.unlink()
    assert reloaded.get("stub:1") is None


def test_cache_survives_a_corrupt_index(tmp_path):
    (tmp_path).mkdir(exist_ok=True)
    (tmp_path / "index.json").write_text("{not json", encoding="utf-8")
    assert visuals.AssetCache(tmp_path).get("stub:1") is None


def test_ledger_cooldown_expires_and_ignores_the_current_run(tmp_path):
    path = tmp_path / "usage.json"
    visuals.UsageLedger(path).record("run-a", ["stub:1"])

    ledger = visuals.UsageLedger(path)
    assert ledger.recently_used("stub:1", cooldown=3, run_id="run-b")
    assert not ledger.recently_used("stub:1", cooldown=0, run_id="run-b")
    assert not ledger.recently_used("stub:1", cooldown=3, run_id="run-a")
    assert not ledger.recently_used("stub:unseen", cooldown=3, run_id="run-b")


def test_ledger_counts_uses_once_per_run(tmp_path):
    path = tmp_path / "usage.json"
    ledger = visuals.UsageLedger(path)
    ledger.record("run-a", ["stub:1"])
    ledger.record("run-a", ["stub:1"])
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["assets"]["stub:1"]["uses"] == 1


# ---------------------------------------------------------------------------
# End to end, against the stub provider
# ---------------------------------------------------------------------------

def test_sourcing_produces_a_usable_manifest(sandbox):
    manifest = run_visuals()

    assert manifest["assets"], "nothing sourced"
    assert manifest["stats"]["beats_failed"] == 0
    assert Path(manifest["manifest_path"]).is_file()
    assert Path(manifest["asset_dir"], "attribution.txt").is_file()

    for asset in manifest["assets"]:
        path = Path(asset["path"])
        assert path.is_absolute() and path.is_file()
        assert (Path(manifest["asset_dir"]) / asset["relative_path"]) == path
        assert asset["attribution"] and asset["source_url"]
        assert asset["role"] in {"primary", "alternate", "spare"}

    on_disk = json.loads(Path(manifest["manifest_path"]).read_text(encoding="utf-8"))
    assert on_disk["run_id"] == manifest["run_id"]


def test_assets_are_ordered_by_beat_and_indexed_consistently(sandbox):
    manifest = run_visuals()
    beat_order = [a["beat_index"] for a in manifest["assets"]]
    assert beat_order == sorted(beat_order)
    for position, asset in enumerate(manifest["assets"]):
        assert asset["index"] == position
    for beat in manifest["beats"]:
        for index in beat["asset_indexes"]:
            assert manifest["assets"][index]["beat_index"] == beat["index"]
        assert visuals.assets_for_beat(manifest, beat["index"]) == [
            manifest["assets"][i] for i in beat["asset_indexes"]
        ]


def test_every_beat_gets_its_own_footage(sandbox):
    manifest = run_visuals(assets_per_beat=1)
    keys = [a["provider_key"] for a in manifest["assets"]]
    assert len(keys) == len(set(keys)), "the same clip was reused inside one video"


def test_identical_files_from_different_ids_are_deduped(sandbox, monkeypatch):
    """Two ids, one file: the second must not be downloaded twice over."""
    StubProvider.catalog = [make_candidate(1), make_candidate(2)]
    session = FakeSession(default=lambda url: FakeResponse(
        200, body=b"identical-bytes" * 32, headers={"Content-Type": "video/mp4"}
    ))
    monkeypatch.setattr(visuals, "_new_session", lambda: session)

    manifest = run_visuals(assets_per_beat=2, headroom=0.1)
    hashes = [a["sha256"] for a in manifest["assets"]]
    assert len(hashes) == len(set(hashes)) == 1


def test_beat_timings_are_scaled_to_the_measured_audio(sandbox):
    manifest = run_visuals(duration_seconds=33.0)
    assert manifest["narration_duration"] == 33.0
    assert manifest["beats"][-1]["end"] == 33.0
    assert manifest["duration_source"] == "caller-supplied"


def test_placement_hints_stay_inside_their_beat(sandbox):
    manifest = run_visuals()
    beats = {b["index"]: b for b in manifest["beats"]}
    for asset in manifest["assets"]:
        beat = beats[asset["beat_index"]]
        if asset["role"] == "spare":
            assert asset["suggested_start"] is None
            continue
        assert beat["start"] - 0.01 <= asset["suggested_start"] <= beat["end"]


def test_headroom_is_honoured(sandbox):
    manifest = run_visuals(headroom=2.0, assets_per_beat=1)
    assert manifest["target_duration"] == pytest.approx(80.0)
    assert manifest["sourced_duration"] >= manifest["target_duration"] or any(
        "less slack" in w for w in manifest["warnings"]
    )


def test_short_headroom_sources_less(sandbox, one_shot_beats):
    lean = run_visuals(headroom=0.2, assets_per_beat=1, run_id="lean")
    generous = run_visuals(headroom=3.0, assets_per_beat=1, run_id="generous")
    assert len(generous["assets"]) > len(lean["assets"])


def test_rerunning_the_same_script_downloads_nothing(sandbox):
    first = run_visuals()
    downloads_after_first = len(sandbox.download_calls)

    second = run_visuals()
    assert second["reused_run"] is True
    assert len(sandbox.download_calls) == downloads_after_first
    assert [a["path"] for a in second["assets"]] == [a["path"] for a in first["assets"]]


def test_refresh_re_sources_the_same_script(sandbox):
    run_visuals()
    refreshed = run_visuals(refresh=True)
    assert refreshed["reused_run"] is False


def test_a_missing_file_invalidates_the_reuse(sandbox):
    first = run_visuals()
    Path(first["assets"][0]["path"]).unlink()
    again = run_visuals()
    assert again["reused_run"] is False
    assert Path(again["assets"][0]["path"]).is_file()


def test_cache_survives_across_runs_of_different_scripts(sandbox, monkeypatch):
    """A second script reuses cached bytes instead of fetching them again."""
    monkeypatch.setattr(config, "VISUAL_REUSE_COOLDOWN_RUNS", 0)
    run_visuals(reuse_cooldown=0)
    downloads = len(sandbox.download_calls)

    other = "Rain falls on granite. The granite gives up its salt. The rivers carry it."
    run_visuals(script=other, run_id="second", reuse_cooldown=0, duration_seconds=15.0)
    assert len(sandbox.download_calls) == downloads, "re-downloaded a cached asset"


def test_recent_footage_is_avoided_in_the_next_video(sandbox):
    one_beat = "The salt stays behind."
    first = run_visuals(script=one_beat, duration_seconds=4.0, assets_per_beat=1,
                        headroom=0.1, run_id="video-one")
    second = run_visuals(script=one_beat, duration_seconds=4.0, assets_per_beat=1,
                         headroom=0.1, run_id="video-two", refresh=True)

    assert first["assets"][0]["provider_key"] != second["assets"][0]["provider_key"]


def test_a_dead_download_url_costs_one_asset_not_the_run(sandbox, monkeypatch):
    def route(url):
        if url.endswith("/0.mp4"):
            return FakeResponse(404, body=b"gone")
        return unique_media(url)

    monkeypatch.setattr(visuals, "_new_session", lambda: FakeSession(default=route))
    manifest = run_visuals()

    assert manifest["assets"]
    assert not any(a["provider_id"] == "video0" for a in manifest["assets"])
    assert any("404" in w for w in manifest["warnings"])


def test_a_beat_with_no_results_is_marked_failed_but_the_run_continues(
    sandbox, monkeypatch
):
    real_gather = visuals._gather_candidates

    def nothing_for_the_first_beat(providers, beat, settings, warnings, **kw):
        # **kw because beat 0 now reaches the fallback-query re-search and the
        # neutral shot, both of which call this with `queries=`. It answers
        # nothing to all of them: the point of this test is a beat that no
        # query can fill, and the run surviving it.
        if beat.index == 0:
            return []
        return real_gather(providers, beat, settings, warnings, **kw)

    monkeypatch.setattr(visuals, "_gather_candidates", nothing_for_the_first_beat)
    manifest = run_visuals()

    assert manifest["assets"], "one dead beat killed the whole run"
    assert manifest["stats"]["beats_failed"] == 1
    failed = [b for b in manifest["beats"] if b["status"] == "failed"]
    assert failed[0]["index"] == 0
    assert failed[0]["notes"] and failed[0]["asset_indexes"] == []
    assert any("beat 0" in w for w in manifest["warnings"])
    # every other beat still got its footage
    assert {b["index"] for b in manifest["beats"] if b["status"] != "failed"} == {
        b["index"] for b in manifest["beats"]
    } - {0}


def test_no_candidate_is_ever_downloaded_twice_in_one_run(sandbox, monkeypatch):
    """The headroom top-up re-walks each beat's leftovers.

    Live, that meant every candidate that had already failed was fetched again
    on every pass - tens of megabytes of traffic for nothing.
    """
    monkeypatch.setattr(visuals.time, "sleep", lambda _: None)
    monkeypatch.setattr(visuals, "_new_session", lambda: FakeSession(
        default=lambda url: FakeResponse(500, body=b"upstream is down")
    ))
    with pytest.raises(visuals.NoAssetsError):
        run_visuals()

    urls = [c["url"] for c in sandbox.download_calls]
    assert urls == [], "session was replaced; nothing should have hit the old one"


def test_a_failed_download_is_not_attempted_again(sandbox):
    dead = "/0.mp4"
    sandbox.routes[dead] = FakeResponse(404, body=b"gone")

    manifest = run_visuals()

    attempts = [c for c in sandbox.download_calls if c["url"].endswith(dead)]
    assert len(attempts) == 1, f"retried a known-dead URL {len(attempts)} times"
    assert manifest["assets"]


def test_every_download_url_is_requested_at_most_once(sandbox):
    manifest = run_visuals(headroom=3.0)
    urls = [c["url"] for c in sandbox.download_calls]
    assert len(urls) == len(set(urls))
    assert manifest["assets"]


def test_download_rate_limit_stops_that_host_for_the_run(sandbox, monkeypatch):
    """Openverse hands back Wikimedia URLs, and Wikimedia does throttle."""
    monkeypatch.setattr(visuals.time, "sleep", lambda _: None)
    session = FakeSession(default=FakeResponse(429, body=b"slow down"))
    monkeypatch.setattr(visuals, "_new_session", lambda: session)

    with pytest.raises(visuals.NoAssetsError):
        run_visuals()

    # one attempt plus one backoff retry, then the host is left alone
    assert len(session.download_calls) == 2


def test_a_beat_cannot_burn_unlimited_downloads(sandbox, monkeypatch):
    attempted = []

    def flaky(url):
        attempted.append(url)
        return FakeResponse(500, body=b"nope")

    monkeypatch.setattr(visuals, "_new_session", lambda: FakeSession(default=flaky))
    with pytest.raises(visuals.NoAssetsError):
        run_visuals()

    assert len(attempted) <= len(StubProvider.catalog)
    assert len(attempted) == len(set(attempted))


def test_the_same_problem_is_only_reported_once(sandbox):
    sandbox.routes["/0.mp4"] = FakeResponse(404, body=b"gone")
    manifest = run_visuals(headroom=3.0)
    assert len(manifest["warnings"]) == len(set(manifest["warnings"]))


def test_asset_filenames_are_unique_including_spares(sandbox, one_shot_beats):
    manifest = run_visuals(headroom=3.0, assets_per_beat=1)
    names = [a["relative_path"] for a in manifest["assets"]]
    assert len(names) == len(set(names))
    assert any(a["role"] == "spare" for a in manifest["assets"])


def test_spares_do_not_all_pile_onto_the_first_beat(sandbox, one_shot_beats):
    """Ties in the top-up ordering used to hand every spare to beat 0."""
    manifest = run_visuals(headroom=3.0, assets_per_beat=1)
    spare_beats = {a["beat_index"] for a in manifest["assets"] if a["role"] == "spare"}
    assert len(spare_beats) > 1, f"all spares landed on {spare_beats}"


def test_rate_limited_provider_is_benched_not_hammered(sandbox):
    StubProvider.failure = visuals.ProviderRateLimitError("slow down", provider="stub")
    with pytest.raises(visuals.NoAssetsError):
        run_visuals()


def test_everything_below_the_floor_means_no_assets(sandbox):
    StubProvider.catalog = [make_candidate(i, width=1280, height=720) for i in range(5)]
    with pytest.raises(visuals.NoAssetsError):
        run_visuals()


def test_rejections_are_counted_for_diagnosis(sandbox):
    StubProvider.catalog = [
        make_candidate(0),
        *[make_candidate(i, width=1280, height=720) for i in range(1, 6)],
    ]
    manifest = run_visuals(assets_per_beat=1)
    rejected = manifest["stats"]["rejected_candidates"]
    assert any("floor" in reason for reason in rejected)


def test_images_can_be_sourced_and_are_measured_after_download(sandbox):
    StubProvider.catalog = [
        make_candidate(i, kind="image", width=1080, height=1920) for i in range(20)
    ]
    manifest = run_visuals(assets_per_beat=1)
    assert manifest["stats"]["by_kind"] == {"image": len(manifest["assets"])}
    assert all(a["width"] == 1080 and a["height"] == 1920 for a in manifest["assets"])


def test_an_image_smaller_than_it_claimed_is_discarded(sandbox, monkeypatch):
    StubProvider.catalog = [make_candidate(1, kind="image", width=2160, height=3840)]
    monkeypatch.setattr(visuals, "_new_session", lambda: FakeSession(
        default=lambda url: FakeResponse(200, body=jpeg_bytes(360, 640),
                                         headers={"Content-Type": "image/jpeg"})
    ))
    with pytest.raises(visuals.NoAssetsError):
        run_visuals()


def test_every_provider_is_asked_the_primary_query_even_when_the_first_delivers(
    sandbox, monkeypatch
):
    """The primary query goes to all providers; the fallback queries do not.

    This used to stop at the first provider that returned a full pool, to save
    Openverse's 200 anonymous requests a day. The saving was real and the cost
    was worse: sufficiency is counted in clips that are *shot* well enough to
    use, which is decided before anything has looked at what they show. So one
    library answering a historical query with well-shot modern footage ended
    the search, and the relevance screen - which can only reorder the pool it
    is handed - had nothing else to promote. On the Inca run that meant
    Openverse, the only provider here indexing museum and archive collections,
    performed 0 searches.

    The quota argument still holds for the *fallback* queries, which is where
    the early exit now lives, so the extra spend is one search per provider per
    beat and only on the query most likely to be answered.
    """

    class FallbackProvider(StubProvider):
        name = "fallback"
        queries: list[str] = []

        def search(self, query, kind, limit):
            FallbackProvider.queries.append(query)
            return []

    FallbackProvider.queries = []
    monkeypatch.setattr(
        visuals, "_PROVIDERS", {"stub": StubProvider, "fallback": FallbackProvider}
    )

    manifest = run_visuals(providers=["stub", "fallback"])

    assert manifest["assets"]
    asked = FallbackProvider.queries
    assert asked, "the primary query never reached the second provider"

    primary = {b["query"] for b in manifest["beats"]}
    assert set(asked) <= primary, (
        f"a fallback query was spent on the second provider: "
        f"{sorted(set(asked) - primary)}"
    )
    # One search per beat: the pool is already full by the time this provider is
    # reached, so the still search for each beat is still skipped.
    assert len(asked) == len(manifest["beats"])


def test_a_fallback_provider_is_used_when_the_first_one_is_empty(
    sandbox, monkeypatch
):
    class EmptyProvider(StubProvider):
        name = "empty"

        def search(self, query, kind, limit):
            self.searches += 1
            return []

    monkeypatch.setattr(
        visuals, "_PROVIDERS", {"empty": EmptyProvider, "stub": StubProvider}
    )
    manifest = run_visuals(providers=["empty", "stub"])
    assert manifest["assets"]
    assert manifest["stats"]["by_provider"] == {"stub": len(manifest["assets"])}


def test_unknown_provider_name_warns_and_is_skipped(sandbox):
    manifest = run_visuals(providers=["stub", "nope"])
    assert any("unknown provider" in w for w in manifest["warnings"])
    assert manifest["assets"]


def test_no_configured_provider_is_a_clear_failure(sandbox):
    with pytest.raises(visuals.NoAssetsError, match="no usable providers"):
        run_visuals(providers=["nope"])


def test_dry_run_plans_without_touching_the_network(sandbox):
    manifest = run_visuals(dry_run=True)
    assert manifest["assets"] == []
    assert sandbox.calls == []
    assert all(b["query"] and b["status"] == "planned" for b in manifest["beats"])
    assert manifest["manifest_path"] is None


def test_step2_and_step3_results_are_accepted_directly(sandbox, tmp_path):
    """The dicts steps 2 and 3 already return should just work."""
    manifest = visuals.source_visuals(
        {"topic": "why the ocean is salty", "script": SCRIPT, "estimated_duration": 40},
        {"audio_path": str(tmp_path / "x.mp3"), "duration_seconds": 37.5,
         "character_count": len(SCRIPT)},
        providers=["stub"], query_mode="heuristic",
    )
    assert manifest["topic"] == "why the ocean is salty"
    assert manifest["narration_duration"] == 37.5
    assert manifest["duration_source"] == "measured (step 3 result)"


def test_empty_script_is_refused_before_anything_happens(sandbox):
    with pytest.raises(ValueError, match="empty"):
        run_visuals(script="   ")
    assert sandbox.calls == []


def test_missing_audio_file_is_reported(sandbox, tmp_path):
    with pytest.raises(ValueError, match="audio file not found"):
        visuals.source_visuals(SCRIPT, tmp_path / "nope.mp3", providers=["stub"],
                               query_mode="heuristic")


@pytest.mark.parametrize("kwargs,message", [
    ({"assets_per_beat": 0}, "at least 1"),
    ({"headroom": 0}, "must be positive"),
])
def test_nonsense_settings_are_refused(sandbox, kwargs, message):
    with pytest.raises(ValueError, match=message):
        run_visuals(**kwargs)


def test_run_id_is_stable_for_the_same_script_and_varies_otherwise():
    first = visuals.make_run_id("the ocean", SCRIPT)
    assert first == visuals.make_run_id("the ocean", SCRIPT)
    assert first != visuals.make_run_id("the ocean", SCRIPT + " And more.")
    assert "/" not in first and " " not in first


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_dry_run_prints_the_plan(sandbox, capsys):
    code = visuals.main([SCRIPT, "--duration", "40", "--dry-run",
                         "--query-mode", "heuristic", "--provider", "stub"])
    printed = capsys.readouterr().out
    assert code == 0
    assert "beat  0" in printed and "query (heuristic)" in printed


def test_cli_sources_and_reports(sandbox, capsys):
    code = visuals.main([SCRIPT, "--duration", "40", "--query-mode", "heuristic",
                         "--provider", "stub", "--assets-per-beat", "1"])
    printed = capsys.readouterr().out
    assert code == 0
    assert "SOURCED" in printed and "attribution" not in printed.lower()[:200]


def test_cli_json_output_is_machine_readable(sandbox, capsys):
    code = visuals.main([SCRIPT, "--duration", "40", "--query-mode", "heuristic",
                         "--provider", "stub", "--json"])
    assert code == 0
    manifest = json.loads(capsys.readouterr().out)
    assert manifest["assets"] and manifest["run_id"]


def test_cli_reports_failure_without_a_traceback(sandbox, capsys):
    code = visuals.main(["", "--duration", "40", "--query-mode", "heuristic"])
    assert code == 1
    assert "error:" in capsys.readouterr().err


def test_cli_providers_status_needs_no_script(capsys):
    assert visuals.main(["--providers-status"]) == 0
    printed = capsys.readouterr().out
    assert "openverse" in printed and "pexels" in printed


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# The opening shot is not scored like the others
# ---------------------------------------------------------------------------

def test_the_opening_beat_takes_a_landscape_clip_over_a_perfect_still():
    """Everywhere else a still is honest filler. On the first shot it is the
    reel announcing that nothing is going to happen, in the second and a half
    where the viewer decides."""
    still = make_candidate(1, kind="image", width=1080, height=1920)
    clip = make_candidate(2, kind="video", width=3840, height=2160, duration=8.0)

    assert visuals.score_candidate(clip, _beat(), provider_rank=0,
                                   recently_used=False, hook=True) \
        > visuals.score_candidate(still, _beat(), provider_rank=0,
                                  recently_used=False, hook=True)


def test_away_from_the_opening_a_vertical_still_still_beats_a_cropped_clip():
    """The premium is paid on the hook and nowhere else - otherwise it would
    just be a global preference dressed up as one."""
    still = make_candidate(1, kind="image", width=1080, height=1920)
    clip = make_candidate(2, kind="video", width=3840, height=2160, duration=8.0)

    ordinary = visuals.score_candidate(still, _beat(), provider_rank=0,
                                       recently_used=False, hook=False)
    assert ordinary > visuals.score_candidate(
        still, _beat(), provider_rank=0, recently_used=False, hook=True
    )


def test_the_opening_prefers_a_clip_that_will_not_have_to_loop():
    """A clip that runs out and restarts snaps back to the frame it opened on,
    which is the one thing the opening shot must not do."""
    covers = make_candidate(1, kind="video", duration=9.0)
    short = make_candidate(2, kind="video", duration=1.0)

    hook_gap = (visuals.score_candidate(covers, _beat(5.0), provider_rank=0,
                                        recently_used=False, hook=True)
                - visuals.score_candidate(short, _beat(5.0), provider_rank=0,
                                          recently_used=False, hook=True))
    plain_gap = (visuals.score_candidate(covers, _beat(5.0), provider_rank=0,
                                         recently_used=False, hook=False)
                 - visuals.score_candidate(short, _beat(5.0), provider_rank=0,
                                           recently_used=False, hook=False))
    assert hook_gap > plain_gap


def test_the_hook_weighting_does_not_disturb_the_default_scoring():
    candidate = make_candidate(1, kind="video")
    assert visuals.score_candidate(candidate, _beat(), provider_rank=0,
                                   recently_used=False) \
        == visuals.score_candidate(candidate, _beat(), provider_rank=0,
                                   recently_used=False, hook=False)


# ---------------------------------------------------------------------------
# Step 2's hook card is carried through to step 5
# ---------------------------------------------------------------------------

def test_the_hook_card_is_read_off_step_2s_result():
    text, topic, card = visuals._resolve_script({
        "script": "An opening line. A body.",
        "topic": "salt",
        "hook_card": "Rain made it salty",
    })
    assert text == "An opening line. A body."
    assert topic == "salt"
    assert card == "Rain made it salty"


def test_a_bare_script_string_carries_no_card():
    text, topic, card = visuals._resolve_script("An opening line. A body.")
    assert text == "An opening line. A body."
    assert (topic, card) == ("", "")


def test_a_step_2_result_without_a_card_is_not_an_error():
    _, _, card = visuals._resolve_script({"script": "An opening. A body.",
                                          "topic": "salt"})
    assert card == ""


# ---------------------------------------------------------------------------
# Similes
#
# A beat that describes one thing by comparing it to another is still about
# the first thing. Measured on a live run, "Enough to crush a submarine like a
# tin can" produced the query 'crushed metal can' and four seconds of spray
# paint cans in the middle of a deep-ocean reel.
# ---------------------------------------------------------------------------

def test_a_comparison_is_read_off_the_sentence():
    got = visuals.simile_terms(
        "Enough to crush a submarine like a tin can. It's utterly dark."
    )
    assert got == {"tin", "can"}


def test_the_comparison_stops_at_the_end_of_its_sentence():
    """Otherwise the rest of the narration is swept in as the comparison."""
    got = visuals.simile_terms("Crushed like a tin can. Bioluminescence follows.")
    assert "bioluminescence" not in got


def test_the_general_stoplist_is_not_used_on_a_comparison():
    """`can` is a modal verb in _STOPWORDS and the whole noun in 'a tin can'."""
    assert "can" in visuals._STOPWORDS
    assert "can" in visuals.simile_terms("crushed like a tin can")


@pytest.mark.parametrize("text, vehicle", [
    ("A squid the size of a bus.", "bus"),
    ("Clouds like cotton wool.", "cotton"),
    ("It moves as if a curtain were drawn.", "curtain"),
    ("Ice no bigger than a marble.", "marble"),
])
def test_the_markers_that_announce_a_comparison(text, vehicle):
    assert vehicle in visuals.simile_terms(text)


def test_a_query_built_on_the_comparison_is_flagged():
    warning = visuals.simile_warning(
        3, "crushed metal can",
        "Enough to crush a submarine like a tin can. It's utterly dark.",
    )
    assert "beat 3" in warning and "'can'" in warning


def test_a_query_about_the_subject_is_left_alone():
    assert not visuals.simile_warning(
        3, "deep sea submarine",
        "Enough to crush a submarine like a tin can. It's utterly dark.",
    )


def test_narration_without_a_comparison_is_never_flagged():
    assert not visuals.simile_warning(
        4, "bioluminescent deep sea", "Below that, creatures make their own light."
    )


def test_step_4_builds_its_clients_with_the_same_deadline(monkeypatch):
    """Step 4 speaks to the SDK itself, so it needs the timeout step 2 has."""
    from google import genai

    from pipeline import script_gen

    monkeypatch.setattr(config, "GEMINI_TIMEOUT_SECONDS", 75.0)
    beats = visuals.segment_script("It is utterly dark down there.", 8.0)
    payload = json.dumps([
        {"index": beat.index, "query": "dark water", "alt": "deep sea",
         "image_prompt": "black water"} for beat in beats
    ])
    built = []

    class FakeModels:
        def generate_content(self, **kwargs):
            return type("R", (), {"text": payload})()

    class RecordingClient:
        def __init__(self, api_key=None, http_options=None):
            built.append(http_options)
            self.models = FakeModels()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(genai, "Client", RecordingClient)

    visuals.llm_queries(beats, "the deep ocean")

    assert built[0] == script_gen.client_http_options()
    assert built[0].timeout == 75_000


def test_the_check_is_advisory_and_does_not_replace_the_query(monkeypatch):
    """A warning the operator can read, not a good query silently discarded."""
    from google import genai

    beats = visuals.segment_script(
        "Enough to crush a submarine like a tin can. It is utterly dark.", 8.0
    )
    payload = json.dumps([
        {"index": beat.index, "query": "crushed metal can",
         "alt": "deep sea pressure", "image_prompt": "a crushed tin can"}
        for beat in beats
    ])

    class FakeModels:
        def generate_content(self, **kwargs):
            return type("R", (), {"text": payload})()

    class FakeClient:
        def __init__(self, api_key=None, http_options=None):
            self.models = FakeModels()
            self.http_options = http_options

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(genai, "Client", FakeClient)

    warnings = visuals.llm_queries(beats, "the deep ocean")

    assert any("comparison" in w for w in warnings)
    assert beats[0].query == "crushed metal can"     # flagged, not overridden
    assert beats[0].query_source == "llm"


def test_the_prompt_tells_the_model_which_half_to_shoot():
    """The warning notices the failure; the prompt is what prevents it."""
    assert "never the comparison" in visuals._QUERY_PROMPT_TEMPLATE
    assert "tin can" in visuals._QUERY_PROMPT_TEMPLATE


def test_the_prompt_puts_the_subject_on_the_first_frame():
    """"The milk in your fridge has less protein than a spider's" was searched
    as milk on 3 of 4 tries, because "than" was not a tell and the sentence
    names the milk first. Replayed with these rules, 3 of 3 asked for the
    spider - on a line the prompt does not quote, so it was not reading the
    answer back."""
    prompt = visuals._QUERY_PROMPT_TEMPLATE
    assert '"than"' in prompt
    assert "Beat 0" in prompt and "cover" in prompt
    assert "spider" not in prompt.lower()


# ---------------------------------------------------------------------------
# Step 2's sidecar
#
# Step 2 writes a .txt and a .json. Only the .json carries the hook card, and
# the documented flow passed the .txt - so the card that step 2 wrote several
# openings to choose between was dropped, and step 5 truncated the first line
# mid-sentence instead ("A Roman messenger could cover 100").
# ---------------------------------------------------------------------------

def test_step_2_json_is_read_as_a_result_not_as_narration():
    raw = json.dumps({"script": "A line. Another line.", "topic": "messengers",
                      "hook_card": "100 miles a day"})
    text, topic, card = visuals._resolve_script(visuals._load_step_2_result(raw))
    assert text == "A line. Another line."
    assert topic == "messengers"
    assert card == "100 miles a day"


def test_a_plain_script_file_is_still_narration():
    raw = "A Roman messenger could cover 100 miles in a day."
    assert visuals._load_step_2_result(raw) == raw


def test_text_that_merely_starts_with_a_brace_is_not_guessed_at():
    raw = "{not json at all, just a script that opens oddly"
    assert visuals._load_step_2_result(raw) == raw


def test_json_that_is_not_a_step_2_result_is_left_alone():
    raw = json.dumps({"something": "else"})
    assert visuals._load_step_2_result(raw) == raw


def test_pexels_is_asked_for_english_descriptions():
    """`alt` is published verbatim as the cover's alt text, and Pexels
    localises it - a reel went out described in Spanish before this."""
    session = FakeSession(routes={"pexels": FakeResponse(200, payload={"videos": []})})
    visuals.PexelsProvider(session).search("ocean waves", "video", 5)

    [call] = session.calls
    assert call["params"]["locale"] == "en-US"


def test_ytdl_console_output_is_swallowed():
    """quiet/no_warnings do not cover yt-dlp's ERROR lines, which otherwise
    land in output/logs looking like a failed run."""
    options = visuals._ytdl_base_options()
    logger = options["logger"]

    for level in ("debug", "info", "warning", "error"):
        getattr(logger, level)("[youtube] xyz: This video is not available")

    assert options["quiet"] is True


def _mole_rat_beat():
    """Beat 3 of the naked mole-rat reel of 2026-09-12, as it ran."""
    return Beat(
        index=3, start=15.055, end=19.222, query="naked mole rat moving slowly",
        text="shifts. Its metabolism crashes. It slows to a crawl.",
        alt_queries=("sleeping naked mole rat",),
        neutral_query="underground dirt tunnels",
    )


def test_the_neutral_shot_is_screened_like_everything_else(monkeypatch):
    """A monastery is not an underground dirt tunnel.

    The neutral query is written to be landscape or material with nobody in
    it, and `_neutral_plan` used to take that as licence not to screen what
    came back. On the naked mole-rat reel of 2026-09-12 beat 3 filled both of
    its slots from the neutral shot "underground dirt tunnels" with a person
    sitting in a boat and Sumela monastery, placed them without anything
    recorded, and the beat finished "ok".
    """
    beat = _mole_rat_beat()
    tunnel = _described(80, "exploring-a-dark-earth-tunnel")
    monastery = _described(81, "sumela-manastiri")
    # Monastery first, so the assertion below is about demotion and not about
    # the order the provider happened to return.
    provider = _query_aware_provider(
        {"underground dirt tunnels": [monastery, tunnel]}
    )(session=None)

    def screen(shortlists, beats, topic, subject_keys=None, establishing=False):
        # The subject test, standing in for the model: it is told the beat was
        # searched for the neutral query, not for the narration's words.
        assert [b.query for b in beats] == ["underground dirt tunnels"]
        verdicts = {}
        for index, candidates in shortlists.items():
            verdicts[index] = {
                position: (True, "") if "tunnel" in (c.source_url or "")
                else (False, "shows a monastery")
                for position, c in enumerate(candidates)
            }
        return verdicts, []

    monkeypatch.setattr(visuals, "llm_screen", screen)

    notes: list[str] = []
    ranked = visuals._neutral_plan(
        beat, [provider], _retry_settings(), visuals.UsageLedger(), "run",
        visuals._RunState(), {}, [], set(),
        topic="how naked mole-rats survive without oxygen", notes=notes,
    )

    assert [row[1].provider_id for row in ranked] == ["v80", "v81"], (
        "the monastery was not demoted below the tunnel"
    )
    assert any("monastery" in note for note in notes), notes


def test_the_neutral_shot_is_judged_as_an_establishing_shot(monkeypatch):
    """Screened, and screened by the rules a fallback should be judged by.

    Screening the neutral shot at all was the 2026-09-12 fix. It was still
    being asked the ordinary question - "is this usable over the narration" -
    whose rules end "a loose match beats a blank screen". A blank screen is
    not what this path is choosing against: the beat has footage, it is one
    clip short, and the honest alternative is to stay short.

    What that costs is measured in `_SCREEN_ESTABLISHING_RULES`: the bison
    reel's neutral shot kept two of its three documented rejects without the
    flag and none with it.
    """
    beat = _mole_rat_beat()
    tunnel = _described(80, "exploring-a-dark-earth-tunnel")
    provider = _query_aware_provider(
        {"underground dirt tunnels": [tunnel]}
    )(session=None)
    seen: dict = {}

    def screen(shortlists, beats, topic, subject_keys=None, establishing=False):
        seen["establishing"] = establishing
        # The narration still goes in. The screen has to know what the clip
        # must not contradict; what changes is which way the close calls go.
        seen["text"] = [b.text for b in beats]
        return {i: {p: (True, "") for p in range(len(c))}
                for i, c in shortlists.items()}, []

    monkeypatch.setattr(visuals, "llm_screen", screen)
    visuals._neutral_plan(
        beat, [provider], _retry_settings(), visuals.UsageLedger(), "run",
        visuals._RunState(), {}, [], set(),
        topic="how naked mole-rats survive without oxygen", notes=[],
    )

    assert seen["establishing"] is True, (
        "the neutral shot was screened by the rules written for a beat's own "
        "footage, which prefer keeping"
    )
    assert seen["text"] == [beat.text]


def _deep_settings(**overrides):
    """A beat wanting two clips, with a shortlist of two, so rows 2+ are deep."""
    base = dict(
        relevance_mode="llm",
        relevance_floor=config.VISUAL_RELEVANCE_FLOOR,
        relevance_shortlist=2,
        assets_per_beat=2,
        reuse_cooldown=0,
        kinds=["video"],
    )
    base.update(overrides)
    return _settings(**base)


def test_a_beat_left_short_by_the_screen_looks_further_down_its_own_pool(monkeypatch):
    """The clips were already ranked and already paid for.

    A beat that kept nothing goes to the re-search. A beat that kept
    *something* went nowhere: it was simply short, and pass 2 made up the
    difference from the neutral shot or from rows nothing had judged. Both
    are guesses made while an average of 29 screened-by-nobody candidates sat
    in the same beat's ranking - mean pool per beat across the 87 beats on
    disk is 32.7, of which 7.6 were ever shown to the screen.

    Beat 4 of the giant squid reel is the case: 7 screened, 1 kept for 2
    slots, 30 rows never looked at, and a neutral shark instead of any of
    them.
    """
    good, bad = _described(1, "a-tunnel"), _described(2, "a-monastery")
    deep = _described(3, "another-tunnel")
    plans = {HADZA.index: [(9.0, good, "q"), (8.0, bad, "q"),
                           (7.0, deep, "q")]}
    calls: list[dict] = []

    def screen(shortlists, beats, topic, subject_keys=None, establishing=False):
        calls.append({i: [c.provider_id for c in cs]
                      for i, cs in shortlists.items()})
        return {i: {p: (c.provider_id != "v2", "" if c.provider_id != "v2"
                        else "shows a monastery")
                    for p, c in enumerate(cs)}
                for i, cs in shortlists.items()}, []

    monkeypatch.setattr(visuals, "llm_screen", screen)
    notes = visuals._screen_beats(
        plans, [HADZA], "topic", [], _deep_settings(),
        visuals.UsageLedger(), "run", visuals._RunState(), {}, [],
    )

    assert len(calls) == 2, "the beat was left one short without looking deeper"
    assert calls[1] == {HADZA.index: ["v3"]}, calls
    # Both keepers in front, the rejected monastery behind them.
    assert [row[1].provider_id for row in plans[HADZA.index]] == [
        "v1", "v3", "v2"], (
        "the newly screened keeper did not land above the earlier reject"
    )
    assert any("screened 1 more candidate" in n
               for n in notes.get(HADZA.index, [])), notes


def test_the_deeper_screen_fills_a_beat_that_would_have_gone_neutral(
    sandbox, monkeypatch
):
    """End to end: the beat takes a screened clip over a generic one.

    The squid reel's beat 4 kept 1 of 7 for 2 slots and reached straight past
    30 unjudged rows of its own for "deep ocean underwater". Here the same
    shape is set up with a stub provider, and the beat should now fill from
    the pool it already had.
    """
    keepers = [make_candidate(1), make_candidate(4)]
    StubProvider.catalog = [keepers[0], make_candidate(2), make_candidate(3),
                            keepers[1], make_candidate(5)]
    # A screen that keeps one clip per slice, so the first slice leaves every
    # beat one short and the second slice is what completes it.
    monkeypatch.setattr(
        visuals, "llm_screen",
        _keep_only({keepers[0].provider_id, keepers[1].provider_id}),
    )

    # A shortlist of 2 against a catalog of 5: rows 2-4 are below it, which
    # is the shape this pass exists for.
    monkeypatch.setattr(config, "VISUAL_RELEVANCE_SHORTLIST", 2)
    manifest = run_visuals(
        relevance_mode="llm", assets_per_beat=2, providers=["stub"],
    )

    opening = manifest["beats"][0]
    used = {manifest["assets"][i]["source_url"] for i in opening["asset_indexes"]}
    # Subset, not equality: this beat is three shots long, so it asks for a
    # third clip once both keepers are in and takes it from the rows below.
    # What the pass is for is that the keepers come first and the neutral shot
    # is never reached, and both of those are still exactly stated.
    assert {c.source_url for c in keepers} <= used, (
        "the beat did not fill from the clips the deeper screen kept"
    )
    assert manifest["stats"]["beats_from_neutral"] == 0, (
        "a beat reached for the neutral shot with screened clips still unused"
    )
    assert any("screened" in note and "below the shortlist" in note
               for note in opening["notes"]), opening["notes"]
    assert not any("neutral shot" in note for note in opening["notes"]), (
        "reached for the neutral shot with screened footage still available"
    )
    assert opening["supply"]["screened"] >= 4, opening["supply"]


def test_a_beat_the_screen_filled_does_not_pay_for_a_second_look(monkeypatch):
    """The deeper screen is for beats that came up short, and costs a call."""
    first, second = _described(1, "a-tunnel"), _described(2, "another-tunnel")
    spare = _described(3, "a-third-tunnel")
    plans = {HADZA.index: [(9.0, first, "q"), (8.0, second, "q"),
                           (7.0, spare, "q")]}
    calls: list[int] = []

    def screen(shortlists, beats, topic, subject_keys=None, establishing=False):
        calls.append(1)
        return {i: {p: (True, "") for p in range(len(cs))}
                for i, cs in shortlists.items()}, []

    monkeypatch.setattr(visuals, "llm_screen", screen)
    visuals._screen_beats(
        plans, [HADZA], "topic", [], _deep_settings(),
        visuals.UsageLedger(), "run", visuals._RunState(), {}, [],
    )

    assert len(calls) == 1, "screened deeper into a beat that was already full"


def test_a_beat_the_screen_emptied_re_searches_rather_than_digging(monkeypatch):
    """Nothing kept means the query landed in the wrong world.

    More rows from the same query are more of the same, so these beats keep
    the remedy they had - the fallback queries - and must not be diverted
    into the deeper screen.
    """
    plans = {HADZA.index: [(9.0, _described(1, "a-mixer"), "q"),
                           (8.0, _described(2, "a-lorry"), "q"),
                           (7.0, _described(3, "a-crane"), "q")]}
    shortlists: list[dict] = []

    def screen(shortlists_, beats, topic, subject_keys=None, establishing=False):
        shortlists.append({i: [c.provider_id for c in cs]
                           for i, cs in shortlists_.items()})
        return _reject_everything(shortlists_, beats, topic)

    monkeypatch.setattr(visuals, "llm_screen", screen)
    visuals._screen_beats(
        plans, [HADZA], "topic", [], _deep_settings(),
        visuals.UsageLedger(), "run", visuals._RunState(), {}, [],
    )

    assert shortlists == [{HADZA.index: ["v1", "v2"]}], (
        "a beat with no keepers was sent to the deeper screen instead of the "
        "re-search"
    )


def test_the_deeper_screen_never_shows_the_model_a_clip_twice(monkeypatch):
    """It reaches below the shortlist, and only for rows nothing has judged."""
    rows = [(9.0 - i, _described(i, f"clip-{i}"), "q") for i in range(6)]
    plans = {HADZA.index: list(rows)}
    seen: list[list[str]] = []

    def screen(shortlists, beats, topic, subject_keys=None, establishing=False):
        for candidates in shortlists.values():
            seen.append([c.provider_id for c in candidates])
        # Keep exactly one per slice, so the beat stays short and asks again.
        return {i: {p: (p == 0, "" if p == 0 else "wrong subject")
                    for p in range(len(cs))}
                for i, cs in shortlists.items()}, []

    monkeypatch.setattr(visuals, "llm_screen", screen)
    visuals._screen_beats(
        plans, [HADZA], "topic", [], _deep_settings(),
        visuals.UsageLedger(), "run", visuals._RunState(), {}, [],
    )

    judged = [pid for slice_ in seen for pid in slice_]
    assert len(judged) == len(set(judged)), f"screened a clip twice: {seen}"
    assert seen == [["v0", "v1"], ["v2", "v3"]], seen


def test_a_neutral_shot_the_screen_skipped_says_so(monkeypatch):
    """The quiet half of the failure this path already has a comment about.

    A failed call warns. A reply that came back keyed to some other beat does
    not: `judged` is empty, the rows are returned unscreened, and they are
    then placed with nothing recorded anywhere. That is the same silence the
    mole-rat monastery shipped behind, arrived at by a different route.
    """
    beat = _mole_rat_beat()
    tunnel = _described(80, "exploring-a-dark-earth-tunnel")
    provider = _query_aware_provider(
        {"underground dirt tunnels": [tunnel]}
    )(session=None)

    def screen(shortlists, beats, topic, subject_keys=None, establishing=False):
        return {999: {0: (True, "")}}, []     # a verdict for nobody's beat

    monkeypatch.setattr(visuals, "llm_screen", screen)
    notes: list[str] = []
    ranked = visuals._neutral_plan(
        beat, [provider], _retry_settings(), visuals.UsageLedger(), "run",
        visuals._RunState(), {}, [], set(),
        topic="how naked mole-rats survive without oxygen", notes=notes,
    )

    assert ranked, "the beat was emptied instead of carrying on unscreened"
    assert any("placed unscreened" in note for note in notes), notes


def test_the_ordinary_screen_is_not_judged_as_an_establishing_shot(monkeypatch):
    """The strict rule belongs to the fallback and must not leak.

    A beat's own footage really is choosing against a blank screen, and
    turning the strictness on everywhere would empty beats for the sake of
    keeping the neutral shot honest.
    """
    seen: list[bool] = []

    def screen(shortlists, beats, topic, subject_keys=None, establishing=False):
        seen.append(establishing)
        return {}, []

    monkeypatch.setattr(visuals, "llm_screen", screen)
    plans = {0: [(9.0, _described(12, "anything"), "q")]}
    visuals._screen_beats(
        plans, [HADZA], "topic", [], _retry_settings(),
        visuals.UsageLedger(), "run", visuals._RunState(), {}, [],
    )

    assert seen and not any(seen), (
        "the beat's own shortlist was screened as an establishing shot"
    )


def test_an_unscreened_neutral_shot_is_still_survivable(monkeypatch):
    """Screening the neutral shot must not be a way to empty it.

    `_screen_beats` and acquisition both treat an empty neutral plan as "fall
    through to the rows the screen never saw", which are worse. So a screen
    that rejects everything here demotes and returns, exactly as the main one
    does - it does not drop.
    """
    beat = _mole_rat_beat()
    provider = _query_aware_provider(
        {"underground dirt tunnels": [_described(82, "a-person-in-a-boat")]}
    )(session=None)
    monkeypatch.setattr(visuals, "llm_screen", _reject_everything)

    ranked = visuals._neutral_plan(
        beat, [provider], _retry_settings(), visuals.UsageLedger(), "run",
        visuals._RunState(), {}, [], set(), topic="naked mole-rats",
    )

    assert [row[1].provider_id for row in ranked] == ["v82"]



# ---------------------------------------------------------------------------
# The archive query - a named thing no stock library holds
#
# The hunger stones reel of 2026-09-30 never showed a hunger stone. Openverse
# answered 0 to every query step 4 wrote for it, and 20 Commons photographs of
# the stones to "Hungerstein". Even found, a landscape still ranks at the
# bottom of a pool only whose top 6 are ever screened.
# ---------------------------------------------------------------------------

_HUNGER_TOPIC = "what are the hunger stones of Europe"


def _still(index: int, title: str, provider: str = "archive") -> Candidate:
    """A Commons photograph as Openverse returns it: big, landscape, a still."""
    return Candidate(
        provider=provider, provider_id=f"i{index}", kind="image",
        width=5472, height=3080,
        download_url=f"https://stub.test/image/{index}.jpg",
        source_url=f"https://commons.wikimedia.org/w/index.php?curid={index}",
        license="CC BY-SA 3.0", title=title,
    )


def _hunger_opening():
    return Beat(index=0, start=0.0, end=3.0, query="European hunger stone",
                text="You are looking at Europe's most desperate climate data.",
                archive_query="Hungerstein")


def _elbe_beat(archive: str = "Hungerstein") -> Beat:
    return Beat(index=4, start=18.2, end=24.8, query="Elbe river hunger stone",
                text="The most famous one, in the Elbe river, has an "
                     "inscription from sixteen sixteen: If you see me, weep.",
                archive_query=archive, shots=2)


class _Stock(visuals.Provider):
    name = "stock"
    kinds = ("video",)
    key_env = ()
    asked: list = []

    def search(self, query, kind, limit):
        _Stock.asked.append(query)
        return [make_candidate(n, provider="stock") for n in range(limit)]


class _Archive(visuals.Provider):
    name = "archive"
    kinds = ("image",)
    key_env = ()
    archive = True
    asked: list = []

    def search(self, query, kind, limit):
        _Archive.asked.append(query)
        if query != "Hungerstein":
            return []                    # every word must match, as on Openverse
        return [_still(n, f"Hungerstein in Decin ({n})") for n in range(3)]


@pytest.fixture
def two_libraries():
    _Stock.asked, _Archive.asked = [], []
    return [_Stock(None), _Archive(None)]


def test_the_query_writer_is_asked_what_the_archive_files_it_under():
    parsed = visuals._parse_query_json(json.dumps([
        {"index": 0, "query": "Elbe river hunger stone", "alt": "a",
         "neutral": "n", "archive": "Hungerstein", "image_prompt": "p"},
        {"index": 1, "query": "cracked dry earth"},
    ]), 2)

    assert parsed[0]["archive"] == "Hungerstein"
    assert parsed[1]["archive"] == ""


def test_the_archive_rule_does_not_quote_the_answer_it_was_written_for():
    """A replay of the hunger stones topic must not be reading "Hungerstein"
    back out of the prompt - the example is a different German stone."""
    assert '"archive"' in visuals._QUERY_PROMPT_TEMPLATE
    assert "Stolperstein" in visuals._QUERY_PROMPT_TEMPLATE
    assert "Hungerstein" not in visuals._QUERY_PROMPT_TEMPLATE


def test_the_archive_answer_reaches_the_beat_and_the_manifest(monkeypatch):
    from google import genai

    beats = visuals.segment_script("Hunger stones surface in drought.", 4.0)
    payload = json.dumps([
        {"index": b.index, "query": "Elbe river hunger stone", "alt": "a",
         "neutral": "riverbed stones", "archive": "Hungerstein",
         "image_prompt": "p"} for b in beats
    ])

    class FakeModels:
        def generate_content(self, **kwargs):
            return type("R", (), {"text": payload})()

    class FakeClient:
        def __init__(self, api_key=None, http_options=None):
            self.models = FakeModels()

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setattr(genai, "Client", FakeClient)

    visuals.llm_queries(beats, _HUNGER_TOPIC)

    assert beats[0].archive_query == "Hungerstein"
    assert beats[0].to_dict()["archive_query"] == "Hungerstein"


def test_the_archive_is_asked_even_when_stock_has_filled_the_pool(
        two_libraries):
    pool = visuals._gather_candidates(two_libraries, _elbe_beat(),
                                      _settings(assets_per_beat=2), [])

    assert "Hungerstein" in _Archive.asked
    assert "Hungerstein" not in _Stock.asked
    archived = [c.provider_id for c, _, q in pool if q == "Hungerstein"]
    assert archived == ["i0", "i1", "i2"]


def test_a_re_search_does_not_ask_the_archive_again(two_libraries):
    visuals._gather_candidates(two_libraries, _elbe_beat(),
                               _settings(assets_per_beat=2), [],
                               queries=("Ancient stone inscription",))

    assert "Hungerstein" not in _Archive.asked


def test_a_beat_that_names_nothing_leaves_the_archive_alone(two_libraries):
    visuals._gather_candidates(two_libraries, _elbe_beat(archive=""),
                               _settings(assets_per_beat=2), [])

    assert "Hungerstein" not in _Archive.asked


def test_archive_hits_are_shown_to_the_screen_however_low_they_rank():
    stock = [(9.0, make_candidate(n), "Elbe river hunger stone")
             for n in range(10)]
    stills = [(2.4, _still(n, f"Hungerstein {n}"), "Hungerstein")
              for n in range(3)]

    head, tail = visuals._shortlist(stock + stills, _elbe_beat(), 6)

    assert head == stock[:6] + stills[:visuals.ARCHIVE_SHORTLIST]
    assert tail == stock[6:] + stills[visuals.ARCHIVE_SHORTLIST:]
    plain, rest = visuals._shortlist(stock + stills, _elbe_beat(archive=""), 6)
    assert plain == stock[:6] and rest == stock[6:] + stills


def _screen_elbe(monkeypatch, plans, verdicts_by_key, opening_clip):
    """Beat 4 behind an opening that has its own subject, so what moves on
    beat 4 is this pass's doing and not `_open_on_the_subject`'s."""
    plans[0] = [(9.0, opening_clip, "European hunger stone")]
    verdicts_by_key.setdefault(opening_clip.key, (True, True))
    monkeypatch.setattr(visuals, "llm_screen", _screen_by_key(verdicts_by_key))
    notes = visuals._screen_beats(
        plans, [_hunger_opening(), _elbe_beat()], _HUNGER_TOPIC,
        [_query_aware_provider({})(session=None)],
        _settings(relevance_mode="llm", relevance_floor=0.15,
                  relevance_shortlist=1, assets_per_beat=1,
                  reuse_cooldown=0, kinds=["video", "image"]),
        visuals.UsageLedger(), "run", visuals._RunState(), {}, [],
    )
    return notes


def test_the_photograph_of_the_thing_leads_a_beat_whose_footage_lacks_it(
        monkeypatch):
    """The 2026-09-30 reel's beat 4 - a snowy stream under "If you see me,
    weep" - replayed with the stone found."""
    stream = _described(80, "snowy-mountain-stream")
    castle = _described(81, "meissen-castle-above-the-elbe")
    stone = _still(7, "Hungerstein in Decin (07)")
    plans = {4: [(9.5, stream, "Elbe river hunger stone"),
                 (9.4, castle, "Elbe river hunger stone"),
                 (2.4, stone, "Hungerstein")]}

    notes = _screen_elbe(monkeypatch, plans, {stone.key: (True, True)},
                         _still(9, "Hungerstein in Pirna"))

    assert [row[1].key for row in plans[4]][:2] == [stone.key, stream.key]
    assert any("archive query 'Hungerstein'" in n for n in notes[4])


def test_stock_footage_of_the_subject_keeps_its_place(monkeypatch):
    """A bison reel keeps its bison video ahead of a bison photograph."""
    stones_on_video = _described(82, "hunger-stones-exposed-in-the-elbe")
    stone = _still(7, "Hungerstein in Decin (07)")
    plans = {4: [(9.5, stones_on_video, "Elbe river hunger stone"),
                 (2.4, stone, "Hungerstein")]}

    notes = _screen_elbe(monkeypatch, plans, {
        stones_on_video.key: (True, True), stone.key: (True, True),
    }, _still(9, "Hungerstein in Pirna"))

    assert plans[4][0][1].key == stones_on_video.key
    assert not any("archive query" in n for n in notes.get(4, []))


@pytest.mark.parametrize("verdict", [(False, True), (True, False)],
                         ids=["rejected", "not-the-subject"])
def test_only_an_approved_photograph_of_the_subject_is_moved(monkeypatch,
                                                              verdict):
    stream = _described(80, "snowy-mountain-stream")
    stone = _still(7, "Hungerstein in Decin (07)")
    plans = {4: [(9.5, stream, "Elbe river hunger stone"),
                 (2.4, stone, "Hungerstein")]}

    _screen_elbe(monkeypatch, plans, {stone.key: verdict},
                 _still(9, "Hungerstein in Pirna"))

    assert plans[4][0][1].key == stream.key



def test_beats_sharing_an_archive_query_are_dealt_different_photographs(
        monkeypatch):
    """The 2026-09-30 re-run: four beats asked for "Hungerstein", all four
    were shown the same two photographs, and the first beat spent both."""
    stones = [_still(n, f"Hungerstein {n}") for n in range(6)]

    def plan(first_clip):
        return ([(9.5, first_clip, "stone carving")]
                + [(2.4, stone, "Hungerstein") for stone in stones])

    carving = Beat(index=6, start=30.0, end=34.0, query="stone carving",
                   text="While most of these carvings are ancient,",
                   archive_query="Hungerstein", shots=2)
    plans = {4: plan(_described(80, "snowy-mountain-stream")),
             6: plan(_described(83, "chisel-on-marble"))}
    shown = {}

    def screen(shortlists, beats, topic, subject_keys=None, establishing=False):
        shown.update({i: [c.key for c in cands]
                      for i, cands in shortlists.items()})
        return _screen_by_key({s.key: (True, True) for s in stones})(
            shortlists, beats, topic, subject_keys, establishing)

    plans[0] = [(9.0, _still(9, "Hungerstein in Pirna"), "European hunger stone")]
    monkeypatch.setattr(visuals, "llm_screen", screen)
    visuals._screen_beats(
        plans, [_hunger_opening(), _elbe_beat(), carving], _HUNGER_TOPIC,
        [_query_aware_provider({})(session=None)],
        _settings(relevance_mode="llm", relevance_floor=0.15,
                  relevance_shortlist=1, assets_per_beat=1,
                  reuse_cooldown=0, kinds=["video", "image"]),
        visuals.UsageLedger(), "run", visuals._RunState(), {}, [],
    )

    elbe = [k for k in shown[4] if k.startswith("archive:")]
    later = [k for k in shown[6] if k.startswith("archive:")]
    assert elbe == [stones[0].key, stones[1].key]
    assert later == [stones[2].key, stones[3].key]
    assert plans[4][0][1].key == stones[0].key
    assert plans[6][0][1].key == stones[2].key



def test_an_accented_archive_name_is_also_asked_without_its_accents():
    """"Hungerstein Děčín" found 1 photograph and "Hungerstein Decin" 20."""
    asked = []

    class Commons(visuals.Provider):
        name = "archive"
        kinds = ("image",)
        key_env = ()
        archive = True

        def search(self, query, kind, limit):
            asked.append(query)
            if query == "Hungerstein Decin":
                return [_still(n, f"Hungerstein in Decin ({n})") for n in range(3)]
            return []

    pool = visuals._gather_candidates(
        [Commons(None)], _elbe_beat(archive="Hungerstein Děčín"),
        _settings(assets_per_beat=2), [])

    assert asked[-2:] == ["Hungerstein Děčín", "Hungerstein Decin"]
    # Filed under the query the beat wrote, which is what `_shortlist` reads.
    assert {q for _, _, q in pool} == {"Hungerstein Děčín"}
    assert len(pool) == 3


def test_an_unaccented_archive_name_is_asked_once(two_libraries):
    visuals._gather_candidates(two_libraries, _elbe_beat(),
                               _settings(assets_per_beat=2), [])

    assert _Archive.asked.count("Hungerstein") == 1


# ---------------------------------------------------------------------------
# A 429 that says when to come back
#
# Wikimedia, where the archive query's photographs live, answers
# "Retry-After: 10" and then lets the same file through a try or two later.
# ---------------------------------------------------------------------------

def _answers(*responses):
    """A session serving `responses` in order, the last one forever."""
    served = list(responses)

    def next_response(url):
        return served.pop(0) if len(served) > 1 else served[0]

    return FakeSession(default=next_response)


def _limited(retry_after=None):
    headers = {"Retry-After": retry_after} if retry_after is not None else {}
    return FakeResponse(429, body=b"Too Many Requests", headers=headers)


def _photo():
    return FakeResponse(200, body=b"\xff\xd8jpeg",
                        headers={"Content-Type": "image/jpeg"})


def test_a_named_wait_is_honoured_and_the_file_still_arrives(tmp_path,
                                                            monkeypatch):
    slept = []
    monkeypatch.setattr(visuals.time, "sleep", slept.append)
    session = _answers(_limited("10"), _limited("10"), _photo())

    result = visuals._download_with_backoff(
        session, _still(1, "Hungerstein in Radebeul (5)"), tmp_path / "a.jpg")

    assert result["content_type"] == "image/jpeg"
    assert slept == [11.0, 11.0]
    assert len(session.download_calls) == 3


def test_a_named_wait_is_honoured_only_so_many_times(tmp_path, monkeypatch):
    monkeypatch.setattr(visuals.time, "sleep", lambda _: None)
    session = _answers(_limited("10"))

    with pytest.raises(visuals.ProviderRateLimitError) as caught:
        visuals._download_with_backoff(
            session, _still(1, "Hungerstein in Decin (07)"), tmp_path / "a.jpg")

    assert caught.value.retry_after == 10.0
    assert len(session.download_calls) == 1 + visuals.DOWNLOAD_RATE_LIMIT_RETRIES


def test_a_long_wait_is_a_refusal_not_something_to_poll(tmp_path, monkeypatch):
    slept = []
    monkeypatch.setattr(visuals.time, "sleep", slept.append)
    session = _answers(_limited("3600"), _photo())

    with pytest.raises(visuals.ProviderRateLimitError):
        visuals._download_with_backoff(
            session, _still(1, "Hungerstein in Decin (07)"), tmp_path / "a.jpg")

    assert slept == []
    assert len(session.download_calls) == 1


def test_a_429_with_no_wait_named_still_gets_its_one_retry(tmp_path,
                                                            monkeypatch):
    slept = []
    monkeypatch.setattr(visuals.time, "sleep", slept.append)
    session = _answers(_limited(), _limited(), _photo())

    with pytest.raises(visuals.ProviderRateLimitError) as caught:
        visuals._download_with_backoff(
            session, _still(1, "Hungerstein in Decin (07)"), tmp_path / "a.jpg")

    assert caught.value.retry_after is None
    assert slept == [visuals.RATE_LIMIT_BACKOFF_SECONDS]
    assert len(session.download_calls) == 2

