"""Step 3 - text-to-speech via the ElevenLabs API.

Takes a narration script (the `script` field produced by step 2) and renders it
to a single .mp3 on disk, reporting the *measured* duration of that file and the
number of characters billed to the ElevenLabs account.

Talks to the REST API with `requests` rather than the `elevenlabs` SDK: this is
one POST to one endpoint, and going direct keeps the provider's own error
payload (`detail.status` / `detail.message`) intact all the way to the caller.

Standalone use:
    python pipeline/tts.py "the words to speak"
    python pipeline/tts.py --script-file output/scripts/ocean.txt
    python pipeline/tts.py "hello" -o output/audio/hello.mp3 --voice <voice_id>
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

# Allow both `python pipeline/tts.py` and `python -m pipeline.tts`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Tunables - change the sound of the voice here, not in the request builder.
# ---------------------------------------------------------------------------

#: ElevenLabs REST base. Versioned by them; unlikely to move.
API_BASE_URL = "https://api.elevenlabs.io/v1"

#: Synthesis model. Overridable via ELEVENLABS_MODEL in .env.
#: eleven_multilingual_v2  - best quality, 10k char/request  (current default)
#: eleven_flash_v2_5       - fastest + cheapest, 40k char/request
#: eleven_v3               - most expressive, 5k char/request
MODEL_ID = config.ELEVENLABS_MODEL

#: Container + sample rate + bitrate. mp3_44100_128 is the API default; higher
#: MP3 tiers need a Creator-plan subscription.
OUTPUT_FORMAT = "mp3_44100_128"

# --- voice_settings ---------------------------------------------------------
# All 0.0-1.0 unless noted. Values below are ElevenLabs' own defaults.

#: Lower = more emotional range and more variation between takes.
#: Higher = flatter, more repeatable. 0.4-0.6 suits narration.
STABILITY = 0.5

#: How hard the model tries to match the original voice's timbre. Pushed too
#: high on a noisy source voice, it reproduces the artefacts too.
SIMILARITY_BOOST = 0.75

#: Style exaggeration. Costs latency and hurts stability; 0 for straight reads.
STYLE = 0.0

#: Slight clarity/likeness bump. Free, so on by default.
USE_SPEAKER_BOOST = True

#: Playback rate. 1.0 = natural. Accepted range is 0.7-1.2.
SPEED = 1.0

# --- request behaviour ------------------------------------------------------

#: Spell out numbers, dates and currency. "on" is more correct but slower and
#: unsupported on flash/turbo models; "auto" lets the API decide per model.
APPLY_TEXT_NORMALIZATION = "auto"

#: (connect, read) seconds. Reads are slow - the whole file streams back.
REQUEST_TIMEOUT = (10, 180)

#: Bytes pulled per iteration while streaming the response to disk.
DOWNLOAD_CHUNK_SIZE = 8192

# --- input guards -----------------------------------------------------------

#: Per-request ceiling. Deliberately below the 10k the current model allows: a
#: reel script this long means something upstream went wrong, and one 5k-char
#: mistake is a meaningful slice of a month's credits.
MAX_SCRIPT_CHARACTERS = 5000

#: Below this there is nothing worth narrating.
MIN_SCRIPT_CHARACTERS = 2


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class TTSError(RuntimeError):
    """Base class for every ElevenLabs failure surfaced by this module."""


class TTSAuthError(TTSError):
    """The API key is missing, malformed, or rejected."""


class TTSQuotaError(TTSError):
    """Billing blocked the request: out of credits, or the plan forbids it."""


class TTSRateLimitError(TTSError):
    """Too many concurrent requests, or the service is shedding load."""


#: detail.status values that mean "no credits left", not "bad key". ElevenLabs
#: returns quota_exceeded under HTTP 401, which reads as an auth failure unless
#: the body is inspected - hence this map.
_QUOTA_STATUSES = frozenset({
    "quota_exceeded",
    "insufficient_credits",
    "max_character_limit_exceeded",
})

_AUTH_STATUSES = frozenset({
    "invalid_api_key",
    "missing_api_key",
    "unauthorized",
    "sign_in_required",
    "detected_unusual_activity",
    "missing_permissions",  # key authenticated but lacks a required scope
})

_RATE_LIMIT_STATUSES = frozenset({
    "rate_limit_exceeded",
    "concurrent_limit_exceeded",
    "too_many_concurrent_requests",
    "system_busy",
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _validate_script(script: str) -> str:
    """Return the exact text to send, or explain why it cannot be sent."""
    if not isinstance(script, str):
        raise ValueError(f"script must be a string, got {type(script).__name__}")

    text = script.strip()
    if not text:
        raise ValueError("script is empty - nothing to narrate")
    if len(text) < MIN_SCRIPT_CHARACTERS:
        raise ValueError(f"script is too short to narrate ({len(text)} characters)")
    if len(text) > MAX_SCRIPT_CHARACTERS:
        raise ValueError(
            f"script is {len(text)} characters, over the "
            f"{MAX_SCRIPT_CHARACTERS}-character per-request limit. Split it "
            f"into several calls, or raise MAX_SCRIPT_CHARACTERS if the model "
            f"({MODEL_ID}) allows it."
        )
    return text


def _require_mutagen():
    """Import mutagen up front, so a missing dep fails before credits burn."""
    try:
        from mutagen.mp3 import MP3
    except ImportError as exc:
        raise TTSError(
            "mutagen is required to measure audio duration. "
            "Install it with: pip install mutagen"
        ) from exc
    return MP3


def _describe_api_error(response) -> tuple[str, str]:
    """Pull (status_slug, human_message) out of an ElevenLabs error response.

    Their bodies are `{"detail": {...}}`, but `detail` is sometimes a bare
    string and sometimes a FastAPI validation list, so all three are handled.
    Falls back to the raw text, so the real message is never dropped.
    """
    try:
        detail = response.json().get("detail")
    except ValueError:
        detail = None

    if isinstance(detail, dict):
        slug = str(detail.get("status") or detail.get("code") or "")
        message = str(detail.get("message") or detail.get("type") or detail)
        return slug, message
    if isinstance(detail, list) and detail:
        parts = [
            str(item.get("msg", item)) if isinstance(item, dict) else str(item)
            for item in detail
        ]
        return "", "; ".join(parts)
    if isinstance(detail, str) and detail:
        return "", detail

    body = (response.text or "").strip()
    return "", body[:400] if body else "(no error body returned)"


def _raise_for_api_error(response, voice_id: str) -> None:
    """Translate a non-200 response into the most specific TTSError available."""
    code = response.status_code
    slug, message = _describe_api_error(response)
    label = f"[{code}{'/' + slug if slug else ''}]"

    if slug in _QUOTA_STATUSES:
        raise TTSQuotaError(f"ElevenLabs quota exhausted {label}: {message}")
    if code == 402:
        # Also covers plan restrictions, e.g. free keys and Voice Library voices.
        raise TTSQuotaError(f"ElevenLabs plan/billing limit {label}: {message}")
    if code == 429 or slug in _RATE_LIMIT_STATUSES:
        raise TTSRateLimitError(f"ElevenLabs rate limited {label}: {message}")
    if code in (401, 403) or slug in _AUTH_STATUSES:
        raise TTSAuthError(
            f"ElevenLabs rejected the API key {label}: {message} "
            "(check ELEVENLABS_API_KEY in .env)"
        )
    if code == 404:
        raise TTSError(
            f"ElevenLabs could not find voice {voice_id!r} {label}: {message}"
        )
    raise TTSError(f"ElevenLabs request failed {label}: {message}")


def _stream_to_file(response, output_path: Path) -> int:
    """Write the response body to `output_path` atomically. Returns bytes."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(output_path.name + ".part")

    written = 0
    try:
        with open(partial, "wb") as handle:
            for chunk in response.iter_content(chunk_size=DOWNLOAD_CHUNK_SIZE):
                if chunk:
                    handle.write(chunk)
                    written += len(chunk)
        if written == 0:
            raise TTSError("ElevenLabs returned a 200 with an empty audio body.")
        os.replace(partial, output_path)  # atomic: never leaves a half file
    except BaseException:
        partial.unlink(missing_ok=True)
        raise

    return written


def read_duration_seconds(audio_path: str | Path) -> float:
    """Measure an .mp3's real duration by parsing its frame headers.

    Uses mutagen: pure Python, no ffmpeg, and it reads headers rather than
    decoding audio - so this is milliseconds regardless of file length.
    """
    MP3 = _require_mutagen()
    try:
        info = MP3(str(audio_path)).info
    except Exception as exc:
        raise TTSError(
            f"Saved {audio_path}, but it could not be parsed as MP3: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    return round(float(info.length), 3)


def _build_payload(text: str) -> dict:
    """Assemble the request body from the constants at the top of this file."""
    return {
        "text": text,
        "model_id": MODEL_ID,
        "apply_text_normalization": APPLY_TEXT_NORMALIZATION,
        "voice_settings": {
            "stability": STABILITY,
            "similarity_boost": SIMILARITY_BOOST,
            "style": STYLE,
            "use_speaker_boost": USE_SPEAKER_BOOST,
            "speed": SPEED,
        },
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_audio(script: str, output_path: str, voice_id: str = None) -> dict:
    """Render `script` to an .mp3 at `output_path` using ElevenLabs.

    Args:
        script: the narration text. Whitespace-stripped before sending.
        output_path: where to write the .mp3. Parent dirs are created.
        voice_id: ElevenLabs voice. Defaults to config.DEFAULT_VOICE_ID.

    Returns:
        {"audio_path": str, "duration_seconds": float, "character_count": int}
        where duration_seconds is measured from the written file and
        character_count is what was billed to the ElevenLabs account.

    Raises:
        ValueError: the script is empty or over MAX_SCRIPT_CHARACTERS.
        config.MissingConfigError: ELEVENLABS_API_KEY is not in .env.
        TTSAuthError / TTSQuotaError / TTSRateLimitError / TTSError: see above.
    """
    import requests

    text = _validate_script(script)
    _require_mutagen()  # fail before spending credits, not after

    api_key = config.require(*config.REQUIRED_TTS)["ELEVENLABS_API_KEY"]
    voice = voice_id or config.DEFAULT_VOICE_ID
    if not voice:
        raise ValueError(
            "No voice_id given and config.DEFAULT_VOICE_ID is unset. "
            "Set ELEVENLABS_VOICE_ID in .env or pass voice_id=..."
        )

    destination = Path(output_path)
    if destination.suffix.lower() != ".mp3":
        raise ValueError(f"output_path must end in .mp3, got {output_path!r}")

    headers = {
        "xi-api-key": api_key,
        "Content-Type": "application/json",
        "Accept": "audio/mpeg",
    }

    try:
        response = requests.post(
            f"{API_BASE_URL}/text-to-speech/{voice}",
            headers=headers,
            params={"output_format": OUTPUT_FORMAT},
            json=_build_payload(text),
            timeout=REQUEST_TIMEOUT,
            stream=True,
        )
    except requests.exceptions.Timeout as exc:
        raise TTSError(
            f"ElevenLabs timed out after {REQUEST_TIMEOUT[1]}s: {exc}"
        ) from exc
    except requests.exceptions.RequestException as exc:
        raise TTSError(
            f"Could not reach ElevenLabs: {type(exc).__name__}: {exc}"
        ) from exc

    with response:
        if response.status_code != 200:
            response.content  # materialise the body before the stream closes
            _raise_for_api_error(response, voice)
        _stream_to_file(response, destination)

    return {
        "audio_path": str(destination.resolve()),
        "duration_seconds": read_duration_seconds(destination),
        "character_count": len(text),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _default_output_path() -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return config.AUDIO_DIR / f"narration_{stamp}.mp3"


def _print_result(result: dict, voice: str) -> None:
    rule = "-" * 68
    size_kb = Path(result["audio_path"]).stat().st_size / 1024
    print(rule)
    print(f"VOICE   : {voice}")
    print(f"MODEL   : {MODEL_ID}")
    print(f"CHARS   : {result['character_count']} (billed)")
    print(f"LENGTH  : {result['duration_seconds']}s (measured from file)")
    print(f"SIZE    : {size_kb:.1f} KB")
    print(f"FILE    : {result['audio_path']}")
    print(rule)
    print("dict:")
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Render a narration script to .mp3 with ElevenLabs."
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("script", nargs="?", help="the text to speak")
    source.add_argument(
        "-f", "--script-file", type=Path,
        help="read the script from a text file instead",
    )
    parser.add_argument(
        "-o", "--output", type=Path, default=None,
        help="output .mp3 path (default: output/audio/narration_<timestamp>.mp3)",
    )
    parser.add_argument(
        "-v", "--voice", default=None,
        help=f"ElevenLabs voice id (default: {config.DEFAULT_VOICE_ID})",
    )
    args = parser.parse_args(argv)

    if args.script_file:
        try:
            script = args.script_file.read_text(encoding="utf-8")
        except OSError as exc:
            print(f"error: could not read {args.script_file}: {exc}", file=sys.stderr)
            return 1
    else:
        script = args.script

    output = args.output or _default_output_path()

    try:
        result = generate_audio(script, str(output), voice_id=args.voice)
    except (ValueError, config.MissingConfigError, TTSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_result(result, args.voice or config.DEFAULT_VOICE_ID)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
