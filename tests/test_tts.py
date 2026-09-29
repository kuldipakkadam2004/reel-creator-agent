"""Tests for pipeline/tts.py.

Everything here is offline: the HTTP call is stubbed and the "audio" is a real
but tiny MP3 built at runtime, so the suite never spends ElevenLabs credits.
For a live end-to-end check run `python pipeline/tts.py "..."` by hand.

    python -m pytest tests/test_tts.py -v
    python tests/test_tts.py          # works without pytest installed
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402
import requests  # noqa: E402

import config  # noqa: E402
from pipeline import tts  # noqa: E402


# ---------------------------------------------------------------------------
# Fixtures / fakes
# ---------------------------------------------------------------------------

def _silent_mp3(seconds: float = 1.0) -> bytes:
    """A valid CBR MP3 of `seconds` silence, assembled from raw frame headers.

    MPEG-1 Layer III, 128 kbps, 44.1 kHz, mono: each frame is 1152 samples =
    26.12 ms, and 417 bytes on the wire. Enough for mutagen to measure.
    """
    header = b"\xff\xfb\x90\x00"          # sync + 128kbps/44.1kHz/mono flags
    frame = header + b"\x00" * (417 - len(header))
    frames = max(1, round(seconds / (1152 / 44100)))
    return frame * frames


class FakeResponse:
    """Minimal stand-in for requests.Response covering what tts.py touches."""

    def __init__(self, status_code=200, body=b"", payload=None, headers=None):
        self.status_code = status_code
        self._payload = payload
        self.content = body if payload is None else json.dumps(payload).encode()
        self.headers = headers or {}
        self.closed = False

    @property
    def text(self):
        return self.content.decode("utf-8", "replace")

    def json(self):
        if self._payload is None:
            raise ValueError("not json")
        return self._payload

    def iter_content(self, chunk_size=8192):
        for i in range(0, len(self.content), chunk_size):
            yield self.content[i:i + chunk_size]

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closed = True
        return False


@pytest.fixture
def captured_post(monkeypatch):
    """Patch requests.post; record the call and return a canned response."""
    calls = []

    def _install(response):
        def fake_post(url, **kwargs):
            calls.append({"url": url, **kwargs})
            return response
        monkeypatch.setattr(requests, "post", fake_post)
        return calls

    return _install


@pytest.fixture
def scripted_post(monkeypatch):
    """Patch requests.post to return a different response per attempt."""
    calls = []

    def _install(*responses):
        def fake_post(url, **kwargs):
            calls.append({"url": url, **kwargs})
            return responses[min(len(calls) - 1, len(responses) - 1)]
        monkeypatch.setattr(requests, "post", fake_post)
        return calls

    return _install


@pytest.fixture(autouse=True)
def _fake_key(monkeypatch):
    """Guarantee a key is present so validation order is what's under test."""
    monkeypatch.setenv("ELEVENLABS_API_KEY", "sk_test_key")


@pytest.fixture(autouse=True)
def _no_real_sleeping(monkeypatch):
    """Record the backoff instead of serving it - the suite stays fast."""
    slept = []
    monkeypatch.setattr(tts.time, "sleep", slept.append)
    return slept


# ---------------------------------------------------------------------------
# Input validation - these must not reach the network
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("bad", ["", "   ", "\n\t  \n"])
def test_empty_script_rejected(bad, tmp_path):
    with pytest.raises(ValueError, match="empty"):
        tts.generate_audio(bad, str(tmp_path / "out.mp3"))


def test_overlong_script_rejected(tmp_path):
    script = "a" * (tts.MAX_SCRIPT_CHARACTERS + 1)
    with pytest.raises(ValueError, match="per-request limit"):
        tts.generate_audio(script, str(tmp_path / "out.mp3"))


def test_max_length_script_accepted(tmp_path, captured_post):
    """Exactly at the limit is fine - the guard is inclusive."""
    captured_post(FakeResponse(200, _silent_mp3(0.5)))
    result = tts.generate_audio(
        "a" * tts.MAX_SCRIPT_CHARACTERS, str(tmp_path / "out.mp3")
    )
    assert result["character_count"] == tts.MAX_SCRIPT_CHARACTERS


def test_non_string_script_rejected(tmp_path):
    with pytest.raises(ValueError, match="must be a string"):
        tts.generate_audio(None, str(tmp_path / "out.mp3"))


def test_non_mp3_output_path_rejected(tmp_path):
    with pytest.raises(ValueError, match="must end in .mp3"):
        tts.generate_audio("hello there", str(tmp_path / "out.wav"))


def test_validation_happens_before_any_http(tmp_path, monkeypatch):
    def explode(*args, **kwargs):
        raise AssertionError("network call made despite invalid input")
    monkeypatch.setattr(requests, "post", explode)
    with pytest.raises(ValueError):
        tts.generate_audio("", str(tmp_path / "out.mp3"))


def test_missing_api_key_raises_config_error(tmp_path, monkeypatch):
    monkeypatch.delenv("ELEVENLABS_API_KEY", raising=False)
    monkeypatch.setattr(
        requests, "post",
        lambda *a, **k: pytest.fail("called API without a key"),
    )
    with pytest.raises(config.MissingConfigError, match="ELEVENLABS_API_KEY"):
        tts.generate_audio("hello there", str(tmp_path / "out.mp3"))


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------

def test_writes_file_and_returns_contract(tmp_path, captured_post):
    captured_post(FakeResponse(200, _silent_mp3(2.0)))
    out = tmp_path / "nested" / "dir" / "narration.mp3"

    result = tts.generate_audio("Hello from the pipeline.", str(out))

    assert out.exists() and out.stat().st_size > 0
    assert set(result) == {"audio_path", "duration_seconds", "character_count"}
    assert Path(result["audio_path"]) == out.resolve()
    assert isinstance(result["duration_seconds"], float)
    assert result["duration_seconds"] == pytest.approx(2.0, abs=0.1)
    assert result["character_count"] == len("Hello from the pipeline.")


def test_character_count_matches_stripped_text(tmp_path, captured_post):
    calls = captured_post(FakeResponse(200, _silent_mp3(0.5)))
    result = tts.generate_audio("  spaced out  ", str(tmp_path / "a.mp3"))

    assert calls[0]["json"]["text"] == "spaced out"
    assert result["character_count"] == len("spaced out")


def test_duration_is_measured_not_estimated(tmp_path, captured_post):
    """Same text, different audio -> different duration. Rules out estimation."""
    captured_post(FakeResponse(200, _silent_mp3(1.0)))
    short = tts.generate_audio("same words", str(tmp_path / "s.mp3"))
    captured_post(FakeResponse(200, _silent_mp3(5.0)))
    long = tts.generate_audio("same words", str(tmp_path / "l.mp3"))

    assert short["character_count"] == long["character_count"]
    assert long["duration_seconds"] > short["duration_seconds"] * 4


def test_request_shape(tmp_path, captured_post):
    calls = captured_post(FakeResponse(200, _silent_mp3(0.5)))
    tts.generate_audio("hi there", str(tmp_path / "a.mp3"), voice_id="voice123")
    call = calls[0]

    assert call["url"] == f"{tts.API_BASE_URL}/text-to-speech/voice123"
    assert call["headers"]["xi-api-key"] == "sk_test_key"
    assert call["headers"]["Accept"] == "audio/mpeg"
    assert call["params"]["output_format"] == tts.OUTPUT_FORMAT
    assert call["stream"] is True
    assert call["timeout"] == tts.REQUEST_TIMEOUT

    body = call["json"]
    assert body["model_id"] == tts.MODEL_ID
    assert body["voice_settings"] == {
        "stability": tts.STABILITY,
        "similarity_boost": tts.SIMILARITY_BOOST,
        "style": tts.STYLE,
        "use_speaker_boost": tts.USE_SPEAKER_BOOST,
        "speed": tts.SPEED,
    }


def test_falls_back_to_config_default_voice(tmp_path, captured_post):
    calls = captured_post(FakeResponse(200, _silent_mp3(0.5)))
    tts.generate_audio("hi there", str(tmp_path / "a.mp3"))
    assert calls[0]["url"].endswith(f"/{config.DEFAULT_VOICE_ID}")


def test_explicit_voice_overrides_default(tmp_path, captured_post):
    calls = captured_post(FakeResponse(200, _silent_mp3(0.5)))
    tts.generate_audio("hi there", str(tmp_path / "a.mp3"), voice_id="custom99")
    assert calls[0]["url"].endswith("/custom99")
    assert config.DEFAULT_VOICE_ID not in calls[0]["url"]


# ---------------------------------------------------------------------------
# API error mapping - the real message must survive
# ---------------------------------------------------------------------------

def test_invalid_api_key(tmp_path, captured_post):
    captured_post(FakeResponse(401, payload={"detail": {
        "status": "invalid_api_key",
        "message": "An invalid API key was supplied.",
    }}))
    with pytest.raises(tts.TTSAuthError) as exc:
        tts.generate_audio("hi there", str(tmp_path / "a.mp3"))
    assert "An invalid API key was supplied." in str(exc.value)
    assert "401/invalid_api_key" in str(exc.value)


def test_quota_exceeded_is_not_reported_as_auth_failure(tmp_path, captured_post):
    """ElevenLabs sends quota_exceeded as a 401 - it must not read as a bad key."""
    captured_post(FakeResponse(401, payload={"detail": {
        "status": "quota_exceeded",
        "message": "You have 12 credits remaining, 340 required.",
    }}))
    with pytest.raises(tts.TTSQuotaError) as exc:
        tts.generate_audio("hi there", str(tmp_path / "a.mp3"))
    assert "12 credits remaining, 340 required" in str(exc.value)
    assert not isinstance(exc.value, tts.TTSAuthError)


def test_rate_limited(tmp_path, captured_post):
    captured_post(FakeResponse(429, payload={"detail": {
        "status": "too_many_concurrent_requests",
        "message": "Too many concurrent requests for your subscription.",
    }}))
    with pytest.raises(tts.TTSRateLimitError, match="Too many concurrent"):
        tts.generate_audio("hi there", str(tmp_path / "a.mp3"))


def test_insufficient_credits_402(tmp_path, captured_post):
    captured_post(FakeResponse(402, payload={"detail": {
        "status": "insufficient_credits", "message": "Out of credits.",
    }}))
    with pytest.raises(tts.TTSQuotaError, match="Out of credits."):
        tts.generate_audio("hi there", str(tmp_path / "a.mp3"))


def test_plan_restriction_402(tmp_path, captured_post):
    """Real response seen from a free-tier key: a plan limit, not empty credits."""
    captured_post(FakeResponse(402, payload={"detail": {
        "status": "payment_required",
        "message": "Free users cannot use library voices via the API.",
    }}))
    with pytest.raises(tts.TTSQuotaError) as exc:
        tts.generate_audio("hi there", str(tmp_path / "a.mp3"))
    assert "Free users cannot use library voices" in str(exc.value)
    assert "plan/billing limit" in str(exc.value)


def test_missing_key_permission_is_an_auth_error(tmp_path, captured_post):
    """Key is valid but lacks a scope - still the user's key to fix."""
    captured_post(FakeResponse(401, payload={"detail": {
        "type": "authentication_error",
        "code": "unauthorized",
        "status": "missing_permissions",
        "message": "The API key you used is missing the permission voices_read.",
    }}))
    with pytest.raises(tts.TTSAuthError, match="missing the permission voices_read"):
        tts.generate_audio("hi there", str(tmp_path / "a.mp3"))


def test_unknown_voice(tmp_path, captured_post):
    captured_post(FakeResponse(404, payload={"detail": {
        "status": "voice_not_found", "message": "A voice with id nope was not found.",
    }}))
    with pytest.raises(tts.TTSError, match="could not find voice 'nope'"):
        tts.generate_audio("hi there", str(tmp_path / "a.mp3"), voice_id="nope")


def test_server_error_body_is_surfaced_verbatim(tmp_path, captured_post):
    captured_post(FakeResponse(500, body=b"upstream exploded"))
    with pytest.raises(tts.TTSError, match="upstream exploded"):
        tts.generate_audio("hi there", str(tmp_path / "a.mp3"))


def test_validation_error_list_body(tmp_path, captured_post):
    """FastAPI-style 422 bodies are a list, not a dict."""
    captured_post(FakeResponse(422, payload={"detail": [
        {"loc": ["body", "text"], "msg": "field required", "type": "value_error"},
    ]}))
    with pytest.raises(tts.TTSError, match="field required"):
        tts.generate_audio("hi there", str(tmp_path / "a.mp3"))


# ---------------------------------------------------------------------------
# Retries - only for failures that never reached the synthesiser
# ---------------------------------------------------------------------------

def test_a_rate_limit_is_retried_and_then_succeeds(tmp_path, scripted_post):
    calls = scripted_post(
        FakeResponse(429, payload={"detail": {
            "status": "too_many_concurrent_requests", "message": "busy",
        }}),
        FakeResponse(200, _silent_mp3(1.0)),
    )
    result = tts.generate_audio("hi there", str(tmp_path / "a.mp3"))

    assert len(calls) == 2
    assert result["duration_seconds"] == pytest.approx(1.0, abs=0.1)


def test_a_server_error_is_retried_up_to_the_limit(tmp_path, captured_post):
    calls = captured_post(FakeResponse(503, body=b"upstream exploded"))
    with pytest.raises(tts.TTSError, match="upstream exploded"):
        tts.generate_audio("hi there", str(tmp_path / "a.mp3"))
    assert len(calls) == tts.MAX_ATTEMPTS


def test_the_retry_sends_exactly_the_same_request(tmp_path, scripted_post):
    calls = scripted_post(
        FakeResponse(500, body=b"blip"),
        FakeResponse(200, _silent_mp3(0.5)),
    )
    tts.generate_audio("hi there", str(tmp_path / "a.mp3"))
    assert calls[0]["json"] == calls[1]["json"]
    assert calls[0]["url"] == calls[1]["url"]


@pytest.mark.parametrize("status,payload", [
    (401, {"detail": {"status": "invalid_api_key", "message": "bad key"}}),
    (401, {"detail": {"status": "quota_exceeded", "message": "no credits"}}),
    (402, {"detail": {"status": "payment_required", "message": "upgrade"}}),
    (404, {"detail": {"status": "voice_not_found", "message": "no voice"}}),
    (422, {"detail": [{"msg": "field required"}]}),
])
def test_a_settled_failure_is_not_retried(status, payload, tmp_path, captured_post):
    """A bad key or an empty balance fails the same way every time."""
    calls = captured_post(FakeResponse(status, payload=payload))
    with pytest.raises(tts.TTSError):
        tts.generate_audio("hi there", str(tmp_path / "a.mp3"))
    assert len(calls) == 1, "retrying this only delays the real message"


def test_a_timeout_is_not_retried(tmp_path, monkeypatch):
    """The generation may have happened and been billed - never pay twice."""
    calls = []

    def timing_out(url, **kwargs):
        calls.append(url)
        raise requests.exceptions.Timeout("read timed out")

    monkeypatch.setattr(requests, "post", timing_out)
    with pytest.raises(tts.TTSError, match="timed out"):
        tts.generate_audio("hi there", str(tmp_path / "a.mp3"))
    assert len(calls) == 1


def test_backoff_grows_between_attempts(tmp_path, captured_post, _no_real_sleeping):
    captured_post(FakeResponse(500, body=b"blip"))
    with pytest.raises(tts.TTSError):
        tts.generate_audio("hi there", str(tmp_path / "a.mp3"))

    assert _no_real_sleeping == [tts.RETRY_BASE_SECONDS, tts.RETRY_BASE_SECONDS * 2]


def test_a_sane_retry_after_is_honoured(tmp_path, scripted_post, _no_real_sleeping):
    scripted_post(
        FakeResponse(429, payload={"detail": {"status": "system_busy"}},
                     headers={"Retry-After": "5"}),
        FakeResponse(200, _silent_mp3(0.5)),
    )
    tts.generate_audio("hi there", str(tmp_path / "a.mp3"))
    assert _no_real_sleeping == [5.0]


def test_an_absurd_retry_after_falls_back_to_the_backoff(
        tmp_path, scripted_post, _no_real_sleeping):
    """A provider asking for ten minutes is not worth blocking the run on."""
    scripted_post(
        FakeResponse(429, payload={"detail": {"status": "system_busy"}},
                     headers={"Retry-After": "600"}),
        FakeResponse(200, _silent_mp3(0.5)),
    )
    tts.generate_audio("hi there", str(tmp_path / "a.mp3"))
    assert _no_real_sleeping == [tts.RETRY_BASE_SECONDS]


def test_an_http_date_retry_after_falls_back_to_the_backoff(
        tmp_path, scripted_post, _no_real_sleeping):
    """Retry-After may be a date; we do not parse those."""
    scripted_post(
        FakeResponse(429, payload={"detail": {"status": "system_busy"}},
                     headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}),
        FakeResponse(200, _silent_mp3(0.5)),
    )
    tts.generate_audio("hi there", str(tmp_path / "a.mp3"))
    assert _no_real_sleeping == [tts.RETRY_BASE_SECONDS]


def test_a_retry_leaves_no_partial_file_behind(tmp_path, scripted_post):
    out = tmp_path / "a.mp3"
    scripted_post(
        FakeResponse(500, body=b"blip"),
        FakeResponse(200, _silent_mp3(0.5)),
    )
    tts.generate_audio("hi there", str(out))
    assert out.exists()
    assert list(tmp_path.glob("*.part")) == []


def test_all_api_errors_share_a_base_class():
    for cls in (tts.TTSAuthError, tts.TTSQuotaError, tts.TTSRateLimitError):
        assert issubclass(cls, tts.TTSError)


# ---------------------------------------------------------------------------
# Transport + file-safety
# ---------------------------------------------------------------------------

def test_timeout_surfaces_as_tts_error(tmp_path, monkeypatch):
    def timeout(*a, **k):
        raise requests.exceptions.Timeout("read timed out")
    monkeypatch.setattr(requests, "post", timeout)
    with pytest.raises(tts.TTSError, match="timed out"):
        tts.generate_audio("hi there", str(tmp_path / "a.mp3"))


def test_connection_error_surfaces_as_tts_error(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise requests.exceptions.ConnectionError("dns failure")
    monkeypatch.setattr(requests, "post", boom)
    with pytest.raises(tts.TTSError, match="Could not reach ElevenLabs"):
        tts.generate_audio("hi there", str(tmp_path / "a.mp3"))


def test_empty_200_body_rejected(tmp_path, captured_post):
    captured_post(FakeResponse(200, b""))
    with pytest.raises(tts.TTSError, match="empty audio body"):
        tts.generate_audio("hi there", str(tmp_path / "a.mp3"))


def test_failed_write_leaves_no_partial_file(tmp_path, captured_post):
    """A non-MP3 200 must not leave a .part turd or a bogus .mp3 behind."""
    captured_post(FakeResponse(200, b"this is not audio at all" * 40))
    out = tmp_path / "a.mp3"
    with pytest.raises(tts.TTSError, match="could not be parsed as MP3"):
        tts.generate_audio("hi there", str(out))
    assert not (tmp_path / "a.mp3.part").exists()


def test_error_response_writes_nothing(tmp_path, captured_post):
    captured_post(FakeResponse(401, payload={"detail": {
        "status": "invalid_api_key", "message": "nope",
    }}))
    out = tmp_path / "a.mp3"
    with pytest.raises(tts.TTSAuthError):
        tts.generate_audio("hi there", str(out))
    assert not out.exists()
    assert list(tmp_path.iterdir()) == []


def test_existing_file_is_overwritten(tmp_path, captured_post):
    out = tmp_path / "a.mp3"
    out.write_bytes(b"stale content")
    captured_post(FakeResponse(200, _silent_mp3(1.5)))
    result = tts.generate_audio("hi there", str(out))
    assert out.read_bytes() != b"stale content"
    assert result["duration_seconds"] == pytest.approx(1.5, abs=0.1)


# ---------------------------------------------------------------------------
# Duration helper
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("seconds", [0.5, 3.0, 10.0])
def test_read_duration_seconds(tmp_path, seconds):
    path = tmp_path / "probe.mp3"
    path.write_bytes(_silent_mp3(seconds))
    assert tts.read_duration_seconds(path) == pytest.approx(seconds, abs=0.1)


def test_read_duration_rejects_garbage(tmp_path):
    path = tmp_path / "junk.mp3"
    path.write_bytes(b"definitely not an mp3")
    with pytest.raises(tts.TTSError, match="could not be parsed as MP3"):
        tts.read_duration_seconds(path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_writes_to_requested_path(tmp_path, captured_post, capsys):
    captured_post(FakeResponse(200, _silent_mp3(1.0)))
    out = tmp_path / "cli.mp3"

    assert tts.main(["hello world", "-o", str(out)]) == 0
    assert out.exists()

    printed = capsys.readouterr().out
    assert str(out.resolve()) in printed
    assert json.loads(printed.split("dict:", 1)[1])["character_count"] == 11


def test_cli_reads_script_file(tmp_path, captured_post, capsys):
    calls = captured_post(FakeResponse(200, _silent_mp3(1.0)))
    script_file = tmp_path / "script.txt"
    script_file.write_text("From a file.\n", encoding="utf-8")

    code = tts.main(["-f", str(script_file), "-o", str(tmp_path / "f.mp3")])

    assert code == 0
    assert calls[0]["json"]["text"] == "From a file."


def test_cli_reports_api_error_and_exits_nonzero(tmp_path, captured_post, capsys):
    captured_post(FakeResponse(401, payload={"detail": {
        "status": "quota_exceeded", "message": "You are out of credits.",
    }}))
    code = tts.main(["hello", "-o", str(tmp_path / "x.mp3")])
    assert code == 1
    assert "You are out of credits." in capsys.readouterr().err


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


def test_the_voice_settings_come_from_config_not_from_constants():
    """They were ElevenLabs' own defaults hard-coded here. Step 9 can only run
    a trial on a setting it can set."""
    assert tts.STABILITY == config.TTS_STABILITY
    assert tts.STYLE == config.TTS_STYLE
    assert tts.SPEED == config.TTS_SPEED
    assert tts.SIMILARITY_BOOST == config.TTS_SIMILARITY_BOOST
    assert tts.USE_SPEAKER_BOOST == config.TTS_SPEAKER_BOOST


# ---------------------------------------------------------------------------
# read_balance - what step 8 asks before it spends anything
# ---------------------------------------------------------------------------

#: The fields /user/subscription returned for this project's key on 2026-09-24,
#: trimmed to the ones read_balance looks at.
_SUBSCRIPTION = {
    "tier": "free",
    "character_count": 7826,
    "character_limit": 10000,
    "can_extend_character_limit": False,
    "allowed_to_extend_character_limit": False,
    "next_character_count_reset_unix": 1791932140,
}


@pytest.fixture
def captured_get(monkeypatch):
    """Patch requests.get; record the call and return a canned response."""
    calls = []

    def _install(response):
        def fake_get(url, **kwargs):
            calls.append({"url": url, **kwargs})
            if isinstance(response, Exception):
                raise response
            return response
        monkeypatch.setattr(requests, "get", fake_get)
        return calls

    return _install


def test_balance_reads_the_subscription_with_the_project_key(captured_get):
    calls = captured_get(FakeResponse(200, payload=_SUBSCRIPTION))

    balance = tts.read_balance()

    assert calls[0]["url"].endswith("/user/subscription")
    assert calls[0]["headers"]["xi-api-key"] == "sk_test_key"
    assert balance["used"] == 7826
    assert balance["limit"] == 10000
    assert balance["remaining"] == 2174
    assert balance["tier"] == "free"
    assert balance["resets"] == datetime.fromtimestamp(1791932140).date()
    assert balance["extendable"] is False


def test_balance_is_extendable_only_when_the_plan_offers_it_and_it_is_on(captured_get):
    offered = {**_SUBSCRIPTION, "can_extend_character_limit": True}
    captured_get(FakeResponse(200, payload=offered))
    assert tts.read_balance()["extendable"] is False

    captured_get(FakeResponse(200, payload={
        **offered, "allowed_to_extend_character_limit": True}))
    assert tts.read_balance()["extendable"] is True


def test_balance_never_goes_negative(captured_get):
    captured_get(FakeResponse(200, payload={
        **_SUBSCRIPTION, "character_count": 10040}))

    assert tts.read_balance()["remaining"] == 0


def test_balance_without_a_reset_date_still_reads(captured_get):
    body = {k: v for k, v in _SUBSCRIPTION.items()
            if k != "next_character_count_reset_unix"}
    captured_get(FakeResponse(200, payload=body))

    assert tts.read_balance()["resets"] is None


def test_balance_refused_is_a_tts_error_carrying_the_message(captured_get):
    captured_get(FakeResponse(401, payload={"detail": {
        "status": "missing_permissions",
        "message": "The API key you used is missing the permission user_read.",
    }}))

    with pytest.raises(tts.TTSError, match="missing the permission user_read"):
        tts.read_balance()


def test_balance_without_the_counts_is_a_tts_error(captured_get):
    captured_get(FakeResponse(200, payload={"tier": "free"}))

    with pytest.raises(tts.TTSError, match="without its character counts"):
        tts.read_balance()


def test_balance_unreachable_is_a_tts_error(captured_get):
    captured_get(requests.exceptions.ConnectionError("no route"))

    with pytest.raises(tts.TTSError, match="Could not reach ElevenLabs"):
        tts.read_balance()
