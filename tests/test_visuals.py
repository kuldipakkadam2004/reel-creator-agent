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
import sys
from hashlib import sha1
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

import config  # noqa: E402
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
    """Point every on-disk location at tmp_path and install the stub provider."""
    monkeypatch.setattr(config, "VISUALS_CACHE_DIR", tmp_path / "cache")
    monkeypatch.setattr(config, "VISUALS_RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(config, "VISUALS_USAGE_FILE", tmp_path / "usage.json")
    monkeypatch.setattr(visuals, "_PROVIDERS", {"stub": StubProvider})

    StubProvider.catalog = [make_candidate(i) for i in range(30)]
    StubProvider.failure = None

    session = FakeSession(default=unique_media)
    monkeypatch.setattr(visuals, "_new_session", lambda: session)
    yield session
    StubProvider.catalog = []
    StubProvider.failure = None


def run_visuals(**kwargs):
    """source_visuals with the arguments every sandbox test wants."""
    params = {
        "script": SCRIPT,
        "duration_seconds": 40.0,
        "topic": "why the ocean is salty",
        "providers": ["stub"],
        "query_mode": "heuristic",
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


@pytest.mark.parametrize("width,height,passes", [
    (1080, 1920, True),      # exactly the frame
    (2160, 3840, True),      # 4K vertical
    (3840, 2160, True),      # 4K landscape survives the crop
    (1920, 1080, False),     # 1080p landscape does not
    (720, 1280, False),      # too small even though the shape is right
    (1000, 1000, False),
])
def test_resolution_floor_is_measured_after_the_crop(width, height, passes):
    assert visuals.meets_resolution_floor(width, height) is passes


def test_floor_can_be_overridden_per_call():
    assert not visuals.meets_resolution_floor(720, 1280)
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
    candidate = make_candidate(1, width=1920, height=1080)
    reason = visuals.reject_reason(candidate, _settings())
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
            {"link": "sd.mp4", "width": 640, "height": 1138},      # under the floor
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
    assert not visuals.meets_resolution_floor(candidate.width, candidate.height)


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


def test_short_headroom_sources_less(sandbox):
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

    def nothing_for_the_first_beat(providers, beat, settings, warnings):
        if beat.index == 0:
            return []
        return real_gather(providers, beat, settings, warnings)

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


def test_asset_filenames_are_unique_including_spares(sandbox):
    manifest = run_visuals(headroom=3.0, assets_per_beat=1)
    names = [a["relative_path"] for a in manifest["assets"]]
    assert len(names) == len(set(names))
    assert any(a["role"] == "spare" for a in manifest["assets"])


def test_spares_do_not_all_pile_onto_the_first_beat(sandbox):
    """Ties in the top-up ordering used to hand every spare to beat 0."""
    manifest = run_visuals(headroom=3.0, assets_per_beat=1)
    spare_beats = {a["beat_index"] for a in manifest["assets"] if a["role"] == "spare"}
    assert len(spare_beats) > 1, f"all spares landed on {spare_beats}"


def test_rate_limited_provider_is_benched_not_hammered(sandbox):
    StubProvider.failure = visuals.ProviderRateLimitError("slow down", provider="stub")
    with pytest.raises(visuals.NoAssetsError):
        run_visuals()


def test_everything_below_the_floor_means_no_assets(sandbox):
    StubProvider.catalog = [make_candidate(i, width=1920, height=1080) for i in range(5)]
    with pytest.raises(visuals.NoAssetsError):
        run_visuals()


def test_rejections_are_counted_for_diagnosis(sandbox):
    StubProvider.catalog = [
        make_candidate(0),
        *[make_candidate(i, width=1920, height=1080) for i in range(1, 6)],
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


def test_a_fallback_provider_is_not_queried_when_the_first_one_delivers(
    sandbox, monkeypatch
):
    """Openverse allows 200 anonymous requests a day; do not burn them for
    nothing when Pexels already returned a full pool."""

    class FallbackProvider(StubProvider):
        name = "fallback"
        searched = 0

        def search(self, query, kind, limit):
            FallbackProvider.searched += 1
            return []

    FallbackProvider.searched = 0
    monkeypatch.setattr(
        visuals, "_PROVIDERS", {"stub": StubProvider, "fallback": FallbackProvider}
    )

    manifest = run_visuals(providers=["stub", "fallback"])

    assert manifest["assets"]
    assert FallbackProvider.searched == 0, "spent a fallback provider's quota"


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
