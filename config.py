"""Central configuration for the Reel Creator Agent.

Single source of truth for:
  * project paths (all derived from this file's location — never absolute)
  * secrets loaded from `.env` via python-dotenv
  * per-step required-key groups, so each module can validate only what it needs

Nothing here performs network calls or side effects beyond creating the
output/asset directories.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Paths — everything is resolved relative to this file so the project can be
# moved to a server without touching any code.
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent

ENV_FILE = BASE_DIR / ".env"

ASSETS_DIR = BASE_DIR / "assets"
VISUALS_DIR = ASSETS_DIR / "visuals"
FONTS_DIR = ASSETS_DIR / "fonts"

#: Step 4 keeps downloaded media in a content-addressed cache and gives every
#: run its own directory of hardlinks into it, so re-runs cost no bytes.
VISUALS_CACHE_DIR = VISUALS_DIR / "cache"
VISUALS_RUNS_DIR = VISUALS_DIR / "runs"

#: Cross-run ledger of which stock asset has been used where, so the same clip
#: does not open every video.
VISUALS_USAGE_FILE = VISUALS_DIR / "usage.json"

OUTPUT_DIR = BASE_DIR / "output"
SCRIPTS_DIR = OUTPUT_DIR / "scripts"
AUDIO_DIR = OUTPUT_DIR / "audio"
#: Step-1 scaffolding, kept for anything that wants a flat drop box. Step 5
#: writes finished reels to OUTPUT_DIR/<date>/<run id>/ instead — see
#: video_run_dir() below — so each reel keeps its captions and attribution
#: together for steps 6 and 7.
VIDEO_DIR = OUTPUT_DIR / "video"
CAPTIONS_DIR = OUTPUT_DIR / "captions"

_MANAGED_DIRS = (
    ASSETS_DIR,
    VISUALS_DIR,
    VISUALS_CACHE_DIR,
    VISUALS_RUNS_DIR,
    FONTS_DIR,
    OUTPUT_DIR,
    SCRIPTS_DIR,
    AUDIO_DIR,
    VIDEO_DIR,
    CAPTIONS_DIR,
)

load_dotenv(ENV_FILE)


class MissingConfigError(RuntimeError):
    """Raised when a required environment variable is absent or empty."""


# ---------------------------------------------------------------------------
# Env access
# ---------------------------------------------------------------------------

def get_env(name: str, default: str | None = None, *, required: bool = False) -> str | None:
    """Read an env var. Blank strings count as unset."""
    value = os.getenv(name, "")
    value = value.strip() if value else ""
    if not value:
        if required:
            raise MissingConfigError(
                f"Missing required environment variable {name!r}. "
                f"Add it to {ENV_FILE.name} (see .env.example)."
            )
        return default
    return value


def require(*names: str) -> dict[str, str]:
    """Assert every named var is set; return them as a dict.

    Reports all missing keys at once instead of failing on the first.
    """
    missing = [n for n in names if not (os.getenv(n) or "").strip()]
    if missing:
        raise MissingConfigError(
            "Missing required environment variable(s): "
            + ", ".join(missing)
            + f". Add them to {ENV_FILE.name} (see .env.example)."
        )
    return {n: os.environ[n].strip() for n in names}


def get_env_int(name: str, default: int) -> int:
    """Read an int env var, falling back to `default` when unset."""
    raw = get_env(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as exc:
        raise MissingConfigError(
            f"{name} must be a whole number, got {raw!r}."
        ) from exc


def get_env_float(name: str, default: float) -> float:
    """Read a float env var, falling back to `default` when unset."""
    raw = get_env(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError as exc:
        raise MissingConfigError(
            f"{name} must be a number, got {raw!r}."
        ) from exc


_TRUTHY = frozenset({"1", "true", "yes", "on", "y"})
_FALSY = frozenset({"0", "false", "no", "off", "n"})


def get_env_bool(name: str, default: bool) -> bool:
    """Read a boolean env var. Accepts 1/true/yes/on and 0/false/no/off."""
    raw = get_env(name)
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in _TRUTHY:
        return True
    if lowered in _FALSY:
        return False
    raise MissingConfigError(
        f"{name} must be a boolean (true/false), got {raw!r}."
    )


def get_env_list(name: str, default: str, *, lower: bool = True) -> tuple[str, ...]:
    """Read a comma-separated env var as a tuple of stripped items.

    Lowercased by default - anything matched against provider names or hashtags
    wants that. Pass lower=False where the case is part of the meaning, such as
    a phrase that gets printed into a caption verbatim.
    """
    raw = get_env(name, default) or ""
    items = (item.strip() for item in raw.split(","))
    return tuple((item.lower() if lower else item) for item in items if item)


def ensure_dirs() -> None:
    """Create the output/asset directories if they don't exist."""
    for directory in _MANAGED_DIRS:
        directory.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Per-step settings. Values are read at import time; required-ness is enforced
# by each module calling `require(*config.REQUIRED_<STEP>)` when it actually
# runs, so importing config never explodes on a half-filled .env.
# ---------------------------------------------------------------------------

# Step 2 — script generation
GEMINI_API_KEY = get_env("GEMINI_API_KEY")
GEMINI_MODEL = get_env("GEMINI_MODEL", "gemini-2.5-flash")
REQUIRED_SCRIPT_GEN = ("GEMINI_API_KEY",)

# Step 3 — text-to-speech
ELEVENLABS_API_KEY = get_env("ELEVENLABS_API_KEY")
ELEVENLABS_MODEL = get_env("ELEVENLABS_MODEL", "eleven_multilingual_v2")

#: "Roger" - an ElevenLabs default voice, verified callable on a free-tier API
#: key. Used only when ELEVENLABS_VOICE_ID is absent from .env, so a fresh
#: clone still speaks. Note: Voice Library voices (and some default ones, e.g.
#: Rachel and Aria) return 402 on free plans - a paid plan lifts that.
_FALLBACK_VOICE_ID = "CwhRBWXzGAHq8TQ4Fs17"

#: Voice used by pipeline/tts.py when no voice_id is passed explicitly.
DEFAULT_VOICE_ID = get_env("ELEVENLABS_VOICE_ID", _FALLBACK_VOICE_ID)

#: Kept as an alias so callers can tell "configured" from "fell back".
ELEVENLABS_VOICE_ID = get_env("ELEVENLABS_VOICE_ID")

#: The voice has a working default, so only the key is genuinely required.
REQUIRED_TTS = ("ELEVENLABS_API_KEY",)

# Step 5 — the frame (9:16 vertical). Declared up here, ahead of step 4,
# because the visual resolution floor derives from the size of the finished
# frame. The rest of step 5's settings are further down.
VIDEO_WIDTH = get_env_int("VIDEO_WIDTH", 1080)
VIDEO_HEIGHT = get_env_int("VIDEO_HEIGHT", 1920)
VIDEO_FPS = get_env_int("VIDEO_FPS", 30)

# --- Step 4 — visual sourcing ----------------------------------------------

#: Providers to query, in preference order. An unconfigured provider is skipped
#: at run time rather than raising, so a missing key degrades the run instead of
#: killing it.
#:   pexels    - video + photo, free key, strongest vertical library
#:   pixabay   - video + photo, free key
#:   openverse - photo only, NO KEY NEEDED (200 req/day anon). Last-resort net.
VISUAL_PROVIDERS = get_env_list("VISUAL_PROVIDERS", "pexels,pixabay,openverse")

#: STOCK_API_KEY was the step-1 placeholder; still honoured as the Pexels key so
#: an .env written before the provider was chosen keeps working.
STOCK_API_KEY = get_env("STOCK_API_KEY")
PEXELS_API_KEY = get_env("PEXELS_API_KEY", STOCK_API_KEY)
PIXABAY_API_KEY = get_env("PIXABAY_API_KEY")

#: Env var(s) each provider can take its key from. Empty tuple = keyless.
#: pipeline/visuals.py reads this to decide which providers are usable.
VISUAL_PROVIDER_KEYS: dict[str, tuple[str, ...]] = {
    "pexels": ("PEXELS_API_KEY", "STOCK_API_KEY"),
    "pixabay": ("PIXABAY_API_KEY",),
    "openverse": (),
}

#: Resolution floor, measured AFTER a centre crop to the 9:16 frame: a 4K
#: landscape clip passes (2160px tall leaves 1215px of width), a 1920x1080 one
#: does not (1080px tall leaves only 608px). Set below VIDEO_* to accept
#: footage that will be upscaled slightly.
VISUAL_MIN_WIDTH = get_env_int("VISUAL_MIN_WIDTH", VIDEO_WIDTH)
VISUAL_MIN_HEIGHT = get_env_int("VISUAL_MIN_HEIGHT", VIDEO_HEIGHT)

#: How many assets to source per visual beat. >1 gives step 5 alternates to cut
#: between without going back to the network.
VISUAL_ASSETS_PER_BEAT = get_env_int("VISUAL_ASSETS_PER_BEAT", 2)

#: Source this multiple of the narration length so step 5 has slack to pace
#: cuts. 1.6 = 60% more footage than there is narration.
VISUAL_DURATION_HEADROOM = get_env_float("VISUAL_DURATION_HEADROOM", 1.6)

#: Stills are allowed as filler; VISUAL_IMAGE_DURATION is how long one is
#: assumed to hold the screen when counting coverage (step 5 decides for real).
VISUAL_ALLOW_IMAGES = get_env_bool("VISUAL_ALLOW_IMAGES", True)
VISUAL_IMAGE_DURATION = get_env_float("VISUAL_IMAGE_DURATION", 4.0)

#: Landscape sources are accepted when they survive the crop-aware floor above.
#: Set false to take only portrait/square originals.
VISUAL_ALLOW_LANDSCAPE = get_env_bool("VISUAL_ALLOW_LANDSCAPE", True)

#: Skip an asset used within this many previous runs, so the same stock clip
#: does not open every video. 0 disables the cooldown.
VISUAL_REUSE_COOLDOWN_RUNS = get_env_int("VISUAL_REUSE_COOLDOWN_RUNS", 3)

#: How search queries are written for each beat.
#:   llm       - one Gemini call turns the beats into visual search queries
#:               (falls back to `heuristic` if the call fails)
#:   heuristic - keyword extraction only, no API call
VISUAL_QUERY_MODE = get_env("VISUAL_QUERY_MODE", "llm")

#: Nothing is unconditionally required: openverse needs no key, so a bare .env
#: still sources visuals. Provider readiness is reported per provider instead.
REQUIRED_VISUALS: tuple[str, ...] = ()

# --- Step 5 — video assembly -----------------------------------------------
# The frame itself is VIDEO_WIDTH/HEIGHT/FPS above (declared early because the
# step-4 resolution floor derives from it). Everything below shapes how the
# footage is paced and how the burned-in captions look — change the look here,
# never in pipeline/assemble.py.

#: ffmpeg/ffprobe locations. Empty = resolve at run time: $PATH first, then the
#: binary shipped by the `imageio-ffmpeg` wheel, so a machine with no system
#: ffmpeg still assembles. ffprobe is optional; without it durations are read
#: from ffmpeg's own stream banner.
FFMPEG_BINARY = get_env("FFMPEG_BINARY")
FFPROBE_BINARY = get_env("FFPROBE_BINARY")

#: Finished reels land in OUTPUT_DIR/<date>/<run id>/ — one directory per reel,
#: holding the .mp4, its captions and the attribution carried over from step 4,
#: so steps 6 and 7 have a single place to read from and add to.
VIDEO_RUN_DATE_FORMAT = get_env("VIDEO_RUN_DATE_FORMAT", "%Y-%m-%d")
VIDEO_FILENAME = get_env("VIDEO_FILENAME", "reel.mp4")

# --- encoding ---------------------------------------------------------------
# Instagram re-encodes whatever it is given, so this targets "clean input to a
# transcoder": high profile H.264, yuv420p, faststart. CRF over a bitrate
# target — the source is stock footage of wildly varying complexity.

VIDEO_CODEC = get_env("VIDEO_CODEC", "libx264")
VIDEO_CRF = get_env_int("VIDEO_CRF", 20)
VIDEO_PRESET = get_env("VIDEO_PRESET", "medium")
VIDEO_PIXEL_FORMAT = get_env("VIDEO_PIXEL_FORMAT", "yuv420p")
AUDIO_CODEC = get_env("AUDIO_CODEC", "aac")
AUDIO_BITRATE = get_env("AUDIO_BITRATE", "192k")
AUDIO_SAMPLE_RATE = get_env_int("AUDIO_SAMPLE_RATE", 44100)

# --- loudness ---------------------------------------------------------------
# Instagram normalises playback to roughly -14 LUFS. ElevenLabs renders at
# whatever level the chosen voice happens to sit at, so without this two reels
# cut a week apart land at audibly different volumes - and a quiet one is
# turned up by the platform along with its noise floor. loudnorm fixes the
# perceived loudness of the narration instead of guessing at a gain.

#: Off leaves the narration exactly as ElevenLabs rendered it.
AUDIO_NORMALIZE = get_env_bool("AUDIO_NORMALIZE", True)

#: Integrated loudness target, LUFS. Instagram's own working point. This is a
#: single-pass loudnorm - it keeps step 5 to one encode, and lands within a
#: couple of LU of the target rather than exactly on it. Measured on a real
#: narration: -24.4 LUFS in, -15.6 LUFS out.
AUDIO_LOUDNESS_TARGET = get_env_float("AUDIO_LOUDNESS_TARGET", -14.0)

#: True-peak ceiling, dBTP. -1.5 leaves headroom for the lossy re-encode
#: Instagram puts the file through, which can push samples above the peak.
AUDIO_LOUDNESS_PEAK = get_env_float("AUDIO_LOUDNESS_PEAK", -1.5)

#: Loudness range, LU. How much quiet-to-loud variation to preserve; a single
#: narrator reading to camera does not have much, so this rarely bites.
AUDIO_LOUDNESS_RANGE = get_env_float("AUDIO_LOUDNESS_RANGE", 11.0)

# --- pacing -----------------------------------------------------------------
# Step 4 decided *which* footage belongs to which stretch of narration; these
# decide how often the picture cuts inside one of those beats.

#: Roughly how long one shot should hold. A beat gets ceil(beat/target) shots,
#: capped by how many assets step 4 actually sourced for it.
SHOT_TARGET_SECONDS = get_env_float("SHOT_TARGET_SECONDS", 3.2)

#: Hard floor and ceiling on a single shot. The floor stops a short beat being
#: chopped into flashes; the ceiling forces a cut when a beat drags.
SHOT_MIN_SECONDS = get_env_float("SHOT_MIN_SECONDS", 1.6)
SHOT_MAX_SECONDS = get_env_float("SHOT_MAX_SECONDS", 6.0)

#: When a clip is much longer than the slot it fills, skip this fraction of it
#: before cutting in — stock footage often opens on a fade or a static frame.
SHOT_SKIP_HEAD_RATIO = get_env_float("SHOT_SKIP_HEAD_RATIO", 0.1)

#: How a shot meets the next one. "cut" is a straight cut (the format's norm);
#: "dip" fades each shot up from and down to black over TRANSITION_SECONDS.
TRANSITION = get_env("TRANSITION", "cut")
TRANSITION_SECONDS = get_env_float("TRANSITION_SECONDS", 0.22)

#: Stills get a slow push-in so they do not read as a frozen video. 1.0 = off.
IMAGE_ZOOM = get_env_float("IMAGE_ZOOM", 1.12)

#: Top and tail of the finished reel, in seconds.
VIDEO_FADE_IN_SECONDS = get_env_float("VIDEO_FADE_IN_SECONDS", 0.4)
VIDEO_FADE_OUT_SECONDS = get_env_float("VIDEO_FADE_OUT_SECONDS", 0.6)
AUDIO_FADE_OUT_SECONDS = get_env_float("AUDIO_FADE_OUT_SECONDS", 0.35)

#: Picture held after the narration stops, so the last word is not cut off by
#: the video ending on the same frame.
VIDEO_TAIL_SECONDS = get_env_float("VIDEO_TAIL_SECONDS", 0.5)

# --- captions ---------------------------------------------------------------
# Burned into the picture with libass. These are the *on-screen* captions; the
# Instagram caption text is step 6 and unrelated.

CAPTIONS_ENABLED = get_env_bool("CAPTIONS_ENABLED", True)

#: Where the word timings come from:
#:   auto       - elevenlabs, then whisper, then estimate (first one available)
#:   elevenlabs - /v1/forced-alignment: the audio plus the known script, so the
#:                words are exact rather than transcribed. Needs the step-3 key.
#:   whisper    - faster-whisper locally, word_timestamps=True. No key, but the
#:                model has to be downloaded once.
#:   estimate   - no network, no model: words are spread across step 4's beat
#:                boundaries (already measured against the .mp3) by syllable
#:                weight. Always available, least accurate.
CAPTION_ALIGNMENT = get_env("CAPTION_ALIGNMENT", "auto")
CAPTION_WHISPER_MODEL = get_env("CAPTION_WHISPER_MODEL", "base")

#: Font for the burned-in captions. CAPTION_FONT_FILE wins if set; otherwise
#: the first font dropped into assets/fonts/ is used; otherwise CAPTION_FONT_NAME
#: is looked up among the installed system fonts. A file inside the project is
#: the portable choice — the reel then looks the same on any machine.
CAPTION_FONT_FILE = get_env("CAPTION_FONT_FILE")
CAPTION_FONT_NAME = get_env("CAPTION_FONT_NAME", "Arial Black")

#: Sized for a 1080x1920 frame and scaled automatically if VIDEO_HEIGHT changes.
CAPTION_FONT_SIZE = get_env_int("CAPTION_FONT_SIZE", 96)

#: One cue is at most this many words / characters — short bursts read better on
#: a phone than full sentences, and keep the highlight moving.
CAPTION_MAX_WORDS = get_env_int("CAPTION_MAX_WORDS", 3)
CAPTION_MAX_CHARS = get_env_int("CAPTION_MAX_CHARS", 24)

#: A cue is held at least this long even if the words were spoken faster.
CAPTION_MIN_CUE_SECONDS = get_env_float("CAPTION_MIN_CUE_SECONDS", 0.45)

#: "highlight" tints the word currently being spoken; "plain" shows the whole
#: cue in one colour.
CAPTION_STYLE = get_env("CAPTION_STYLE", "highlight")
CAPTION_UPPERCASE = get_env_bool("CAPTION_UPPERCASE", True)

#: #RRGGBB. Converted to ASS's &HBBGGRR by pipeline/assemble.py.
CAPTION_PRIMARY_COLOR = get_env("CAPTION_PRIMARY_COLOR", "#FFFFFF")
CAPTION_HIGHLIGHT_COLOR = get_env("CAPTION_HIGHLIGHT_COLOR", "#FFE14D")
CAPTION_OUTLINE_COLOR = get_env("CAPTION_OUTLINE_COLOR", "#000000")

CAPTION_OUTLINE_WIDTH = get_env_float("CAPTION_OUTLINE_WIDTH", 6.0)
CAPTION_SHADOW_DEPTH = get_env_float("CAPTION_SHADOW_DEPTH", 3.0)

#: Distance from the bottom of the frame, in 1080x1920 pixels. ~420 keeps the
#: text clear of Instagram's own caption/button overlay along the bottom.
CAPTION_MARGIN_V = get_env_int("CAPTION_MARGIN_V", 420)

#: Left/right margin, so long cues wrap instead of running to the frame edge.
CAPTION_MARGIN_H = get_env_int("CAPTION_MARGIN_H", 90)

#: Nothing is required: captions fall back to `estimate` with no key at all, and
#: ffmpeg comes from the imageio-ffmpeg wheel if the system has none.
REQUIRED_ASSEMBLE: tuple[str, ...] = ()


def video_run_dir(run_id: str, when=None) -> Path:
    """Directory for one finished reel: output/<date>/<run id>/.

    Dated so a day's output stays together, and keyed by step 4's run id so the
    video sits next to the visuals it was cut from. Not created here — the
    caller makes it, so a dry run leaves no empty directories behind.
    """
    from datetime import date as _date

    stamp = (when or _date.today()).strftime(VIDEO_RUN_DATE_FORMAT)
    return OUTPUT_DIR / stamp / run_id


# --- Step 6 — Instagram post text (caption + hashtags) ---------------------
# The copy that goes in Instagram's caption field. Nothing here touches the
# burned-in on-screen captions above - those are CAPTION_* / step 5. Change how
# every post reads from here, never inside pipeline/caption.py.

#: llm       - Gemini writes the caption and tags from the script (falls back
#:             to `heuristic` by itself if the call fails)
#: heuristic - lines lifted from the script + keyword tags, no API call
POST_MODE = get_env("POST_MODE", "llm")

#: Instagram's hard caption limit - hashtags, line breaks and all.
POST_MAX_CHARS = get_env_int("POST_MAX_CHARS", 2200)

#: Only the opening of a caption survives before Instagram cuts it off with
#: "... more", so the first line has to read as a finished thought on its own.
#: 125 is where the feed truncates; the hook is built to fit inside it and is
#: never left mid-sentence.
POST_HOOK_MAX_CHARS = get_env_int("POST_HOOK_MAX_CHARS", 125)

#: The voice of the post copy. One knob, every caption.
POST_TONE = get_env(
    "POST_TONE",
    "curious and plainspoken - short lines, concrete detail, "
    "no hype stacking and no engagement bait",
)

#: What the closing line should do. Blank omits the CTA altogether.
POST_CTA = get_env(
    "POST_CTA",
    "ask one question the post genuinely wants answered in the replies",
)

#: The account's subject area, used to steer the broad end of the hashtag mix
#: when the script alone is too specific to place. Blank = infer from the script.
POST_NICHE = get_env("POST_NICHE", "")

#: Emoji read as native on Instagram, but a caption studded with them reads as
#: a brand account. A couple, at most.
POST_ALLOW_EMOJI = get_env_bool("POST_ALLOW_EMOJI", True)
POST_MAX_EMOJI = get_env_int("POST_MAX_EMOJI", 2)

#: How many hashtags to land on. Instagram allows 30 and Instagram's own advice
#: is 3-5; a wall of 30 reads as spam and dilutes what the post is classified
#: as. 8 is the default: enough surface for the recommender to place the reel,
#: well short of looking desperate.
POST_HASHTAG_COUNT = get_env_int("POST_HASHTAG_COUNT", 8)

#: Platform hard cap, not a preference. Instagram rejects captions above this.
POST_HASHTAG_LIMIT = 30

#: caption - hashtags close the caption (what most accounts do)
#: comment - kept out of the caption text and handed to step 7 separately, to
#:           be posted as the first comment
POST_HASHTAG_PLACEMENT = get_env("POST_HASHTAG_PLACEMENT", "caption")

#: Tags added to every post whatever the topic - a brand or series tag. These
#: are kept even when they push the generated ones out.
POST_HASHTAGS_ALWAYS = get_env_list("POST_HASHTAGS_ALWAYS", "")

#: Tags never to use. The default list is engagement bait that Instagram
#: quietly limits reach on - it is not a moderation list.
POST_HASHTAGS_BANNED = get_env_list(
    "POST_HASHTAGS_BANNED",
    "follow4follow,followforfollow,like4like,likeforlike,f4f,l4l,"
    "tagsforlikes,followback,followme",
)

#: Tags used by the last N runs are passed to the generator as "recently used"
#: so consecutive posts do not carry an identical block. Relevance still wins -
#: a tag that genuinely belongs is allowed through. 0 disables the ledger.
POST_HASHTAG_HISTORY_RUNS = get_env_int("POST_HASHTAG_HISTORY_RUNS", 5)

#: Cross-run ledger backing the setting above.
POST_HASHTAG_USAGE_FILE = CAPTIONS_DIR / "hashtag_usage.json"

#: Words that must never appear in the copy. Checked after generation, not just
#: asked for in the prompt.
POST_BANNED_WORDS = get_env_list("POST_BANNED_WORDS", "")

#: Phrases every caption must contain - a handle, a disclaimer. Appended
#: verbatim (case intact) if the generated copy does not already carry them.
POST_MUST_INCLUDE = get_env_list("POST_MUST_INCLUDE", "", lower=False)

#: What to do with step 4's per-asset credits, carried into the run directory
#: by step 5:
#:   required - credit only what the licence demands (CC-BY and friends).
#:              Pexels and Pixabay do not, so most runs add nothing.
#:   all      - also add a courtesy "via Pexels" line for the rest
#:   off      - never touch the caption
POST_ATTRIBUTION = get_env("POST_ATTRIBUTION", "required")
POST_ATTRIBUTION_HEADER = get_env("POST_ATTRIBUTION_HEADER", "Footage:")

#: Ceiling on individual credit lines, so a 15-shot reel does not bury the
#: copy. Credits a licence actually requires are never dropped to meet it - if
#: there are more of them than this, they all go in and a warning is raised
#: instead, because dropping one would breach the licence.
POST_ATTRIBUTION_MAX = get_env_int("POST_ATTRIBUTION_MAX", 6)

#: Nothing is unconditionally required: `llm` reuses step 2's GEMINI_API_KEY and
#: degrades to `heuristic` without it.
REQUIRED_POST: tuple[str, ...] = ()


# Step 7 — Telegram delivery
TELEGRAM_BOT_TOKEN = get_env("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = get_env("TELEGRAM_CHAT_ID")
REQUIRED_TELEGRAM = ("TELEGRAM_BOT_TOKEN", "TELEGRAM_CHAT_ID")


ensure_dirs()


# ---------------------------------------------------------------------------
# Self-check: `python config.py`
# ---------------------------------------------------------------------------

def _mask(value: str | None) -> str:
    if not value:
        return "(not set)"
    return f"{value[:4]}...{value[-2:]} ({len(value)} chars)" if len(value) > 8 else "set"


def _report() -> None:
    print(f"BASE_DIR         : {BASE_DIR}")
    print(f".env present     : {ENV_FILE.exists()}")
    print("\nDirectories")
    for directory in _MANAGED_DIRS:
        rel = directory.relative_to(BASE_DIR)
        print(f"  {'ok ' if directory.is_dir() else 'MISSING'} {rel}")

    groups = {
        "Step 2 script gen": REQUIRED_SCRIPT_GEN,
        "Step 3 tts": REQUIRED_TTS,
        "Step 4 visuals": ("PEXELS_API_KEY", "PIXABAY_API_KEY",
                           "STOCK_API_KEY"),
        "Step 6 post text": ("GEMINI_API_KEY",),
        "Step 7 telegram": REQUIRED_TELEGRAM,
    }
    print("\nSecrets")
    for label, keys in groups.items():
        for key in keys:
            print(f"  {label:<18} {key:<22} {_mask(get_env(key))}")

    print("\nStep 4 providers (in preference order)")
    for name in VISUAL_PROVIDERS:
        needed = VISUAL_PROVIDER_KEYS.get(name)
        if needed is None:
            state = "unknown provider name"
        elif not needed:
            state = "ready (no key required)"
        else:
            have = next((k for k in needed if get_env(k)), None)
            state = (f"ready (via {have})" if have
                     else "needs " + " or ".join(needed))
        print(f"  {name:<12} {state}")

    print("\nNon-secret settings")
    for key, value in (
        ("GEMINI_MODEL", GEMINI_MODEL),
        ("ELEVENLABS_MODEL", ELEVENLABS_MODEL),
        ("DEFAULT_VOICE_ID", f"{DEFAULT_VOICE_ID}"
                             f"{'' if ELEVENLABS_VOICE_ID else '  (fallback)'}"),
        ("VIDEO", f"{VIDEO_WIDTH}x{VIDEO_HEIGHT} @ {VIDEO_FPS}fps"),
        ("VISUAL_QUERY_MODE", VISUAL_QUERY_MODE),
        ("VISUAL_MIN_RESOLUTION",
         f"{VISUAL_MIN_WIDTH}x{VISUAL_MIN_HEIGHT} (after 9:16 crop)"),
        ("VISUAL_ASSETS_PER_BEAT", VISUAL_ASSETS_PER_BEAT),
        ("VISUAL_DURATION_HEADROOM",
         f"{VISUAL_DURATION_HEADROOM}x narration"),
        ("VISUAL_REUSE_COOLDOWN_RUNS", VISUAL_REUSE_COOLDOWN_RUNS),
        ("SHOT_TARGET_SECONDS", f"{SHOT_TARGET_SECONDS}s per shot "
                                f"({SHOT_MIN_SECONDS}-{SHOT_MAX_SECONDS}s)"),
        ("TRANSITION", f"{TRANSITION} ({TRANSITION_SECONDS}s)"),
        ("AUDIO_NORMALIZE",
         f"{AUDIO_LOUDNESS_TARGET:g} LUFS, peak {AUDIO_LOUDNESS_PEAK:g} dBTP"
         if AUDIO_NORMALIZE else "off (narration left as rendered)"),
        ("CAPTION_ALIGNMENT", CAPTION_ALIGNMENT),
        ("CAPTION_STYLE", f"{CAPTION_STYLE}, {CAPTION_FONT_NAME} "
                          f"{CAPTION_FONT_SIZE}px, "
                          f"<={CAPTION_MAX_WORDS} words/cue"),
        ("CAPTION_FONT_FILE", CAPTION_FONT_FILE or "(auto-resolved)"),
        ("VIDEO_RUN_DIR", f"output/<{VIDEO_RUN_DATE_FORMAT}>/<run id>/"
                          f"{VIDEO_FILENAME}"),
        ("POST_MODE", POST_MODE),
        ("POST_HASHTAGS", f"{POST_HASHTAG_COUNT} tags in the "
                          f"{POST_HASHTAG_PLACEMENT}"
                          + (f", always {' '.join('#' + t for t in POST_HASHTAGS_ALWAYS)}"
                             if POST_HASHTAGS_ALWAYS else "")),
        ("POST_LIMITS", f"{POST_MAX_CHARS} chars, hook <= {POST_HOOK_MAX_CHARS}"),
        ("POST_ATTRIBUTION", POST_ATTRIBUTION),
    ):
        print(f"  {key:<22} {value}")


if __name__ == "__main__":
    _report()
