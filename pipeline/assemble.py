"""Step 5 - video assembly.

Takes the narration .mp3 from step 3 and the sourced footage from step 4 and
cuts them into one finished 9:16 reel with burned-in captions.

What happens, in order:

  1. step 4's manifest is read as given - its beat order and per-beat timings
     are the edit. Nothing here re-reads the script to work out segmentation;
  2. each beat's window is divided into shots of roughly SHOT_TARGET_SECONDS,
     capped by how many assets that beat actually got. Clips are trimmed to
     their slot, looped when they are too short, and stills get a slow push-in;
  3. the narration is aligned to word-level timestamps (ElevenLabs' forced
     alignment by default - the script is already known, so the words are exact
     rather than transcribed), grouped into short cues and written as ASS;
  4. one ffmpeg pass scales/crops every shot to the frame, concatenates them,
     burns the captions in with libass and muxes the narration;
  5. the reel, its captions and step 4's attribution are written to
     output/<date>/<run id>/, which is where steps 6 and 7 will look.

Every knob - resolution, pacing, font, caption colours - lives in config.py.

Standalone use:
    python pipeline/assemble.py -a output/audio/ocean.mp3 -v assets/visuals/runs/ocean-1234
    python pipeline/assemble.py -a ... -v ... --dry-run
    python pipeline/assemble.py -a ... -v ... --alignment estimate --no-captions
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from pathlib import Path

# Allow both `python pipeline/assemble.py` and `python -m pipeline.assemble`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

# ---------------------------------------------------------------------------
# Tunables. The look (resolution, font, colours, pacing) is config.py's job -
# what follows is mechanical: where fonts live on each OS, how cues are broken
# up, and how long we are willing to wait for ffmpeg.
# ---------------------------------------------------------------------------

#: Directories searched for CAPTION_FONT_NAME when no font file is configured
#: and assets/fonts/ is empty.
_SYSTEM_FONT_DIRS = {
    "win32": (Path(os.environ.get("WINDIR", r"C:\Windows")) / "Fonts",
              Path(os.environ.get("LOCALAPPDATA", ".")) / "Microsoft/Windows/Fonts"),
    "darwin": (Path("/System/Library/Fonts"), Path("/Library/Fonts"),
               Path.home() / "Library/Fonts"),
    "linux": (Path("/usr/share/fonts"), Path("/usr/local/share/fonts"),
              Path.home() / ".local/share/fonts"),
}

#: Common display faces, so the usual CAPTION_FONT_NAME values resolve to a file
#: without opening a few hundred fonts. Purely an optimisation - a name that is
#: not listed still gets found by the scan below.
_FONT_FILE_HINTS = {
    "arial black": ("ariblk.ttf", "Arial Black.ttf"),
    "impact": ("impact.ttf", "Impact.ttf"),
    "segoe ui black": ("seguibl.ttf",),
    "arial bold": ("arialbd.ttf",),
    "verdana bold": ("verdanab.ttf",),
    "helvetica": ("Helvetica.ttc",),
    "dejavu sans bold": ("DejaVuSans-Bold.ttf",),
}

_FONT_SUFFIXES = (".ttf", ".otf", ".ttc")

#: The reference frame the caption numbers in config.py are expressed in. ASS
#: carries these as PlayResX/PlayResY and libass scales the result to whatever
#: VIDEO_WIDTH x VIDEO_HEIGHT actually is, so changing the output resolution
#: does not change how big the captions look.
CAPTION_REFERENCE_WIDTH = 1080
CAPTION_REFERENCE_HEIGHT = 1920

#: A pause longer than this between two words ends the cue - that is a breath or
#: a full stop, and holding the previous words across it reads as lag.
CUE_SPLIT_GAP_SECONDS = 0.55

#: A cue is held until the next one starts when the gap is no bigger than this,
#: so short pauses do not blink the captions off and on.
CUE_HOLD_GAP_SECONDS = 0.7

#: Otherwise it lingers this long past its last word.
CUE_LINGER_SECONDS = 0.25

#: Word endings that close a cue even when it is still under the word limit.
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*$")

#: How many characters of ffmpeg's stderr to quote when it fails. Enough for the
#: real error, which ffmpeg prints last.
FFMPEG_ERROR_TAIL = 1800

#: Seconds. Encoding a 60s reel takes well under a minute on a laptop CPU; this
#: is a runaway guard, not a target.
FFMPEG_TIMEOUT = 1800

#: A filter graph longer than this goes into a file (-filter_complex_script)
#: rather than onto the command line, which Windows caps at 32k characters.
FILTER_GRAPH_INLINE_LIMIT = 8000

#: (connect, read) seconds for the ElevenLabs alignment call. The upload is the
#: slow half - the whole .mp3 goes over the wire.
ALIGNMENT_TIMEOUT = (15, 300)

ALIGNMENT_URL = "https://api.elevenlabs.io/v1/forced-alignment"

#: Filenames inside the output run directory.
CAPTIONS_ASS_NAME = "captions.ass"
CAPTIONS_SRT_NAME = "captions.srt"
ALIGNMENT_NAME = "alignment.json"
ATTRIBUTION_NAME = "attribution.txt"
ASSEMBLY_NAME = "assembly.json"

ASSEMBLY_VERSION = 1


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class AssemblyError(RuntimeError):
    """Base class for every failure surfaced by this module."""


class FFmpegNotFoundError(AssemblyError):
    """No ffmpeg binary could be located."""


class FFmpegError(AssemblyError):
    """ffmpeg ran and failed. Carries the tail of its stderr."""


class NoFootageError(AssemblyError):
    """Step 4's manifest yielded no usable file to put on screen."""


# ---------------------------------------------------------------------------
# ffmpeg discovery and probing
# ---------------------------------------------------------------------------

_BINARY_CACHE: dict[str, str | None] = {}


def _find_binary(name: str, configured: str | None) -> str | None:
    """Locate ffmpeg/ffprobe: config first, then $PATH, then imageio-ffmpeg."""
    if name in _BINARY_CACHE:
        return _BINARY_CACHE[name]

    found: str | None = None
    if configured:
        candidate = Path(configured).expanduser()
        found = str(candidate) if candidate.is_file() else shutil.which(configured)
        if not found:
            raise FFmpegNotFoundError(
                f"{name.upper()}_BINARY is set to {configured!r}, which is not "
                f"an executable. Fix it in .env, or unset it to auto-detect."
            )
    if not found:
        found = shutil.which(name)
    if not found and name == "ffmpeg":
        try:
            import imageio_ffmpeg

            found = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:  # not installed, or no binary for this platform
            found = None

    _BINARY_CACHE[name] = found
    return found


def ffmpeg_binary() -> str:
    """Path to ffmpeg, or an error explaining how to get one."""
    found = _find_binary("ffmpeg", config.FFMPEG_BINARY)
    if not found:
        raise FFmpegNotFoundError(
            "ffmpeg not found. Install it and put it on PATH, set FFMPEG_BINARY "
            "in .env, or `pip install imageio-ffmpeg` to use the bundled build."
        )
    return found


def ffprobe_binary() -> str | None:
    """Path to ffprobe if this machine has one. Optional - see `probe_media`."""
    return _find_binary("ffprobe", config.FFPROBE_BINARY)


_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d\d):(\d\d(?:\.\d+)?)")
_VIDEO_STREAM_RE = re.compile(r"Stream #\d+:\d+.*?: Video:.*?(\d{2,5})x(\d{2,5})")


def _probe_with_ffprobe(path: Path) -> dict | None:
    """Exact stream facts from ffprobe, when the machine has one."""
    exe = ffprobe_binary()
    if not exe:
        return None
    args = [exe, "-v", "error", "-show_entries",
            "format=duration:stream=codec_type,width,height,duration",
            "-of", "json", str(path)]
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=60)
        payload = json.loads(done.stdout or "{}")
    except (OSError, ValueError, subprocess.SubprocessError):
        return None

    streams = payload.get("streams") or []
    video = next((s for s in streams if s.get("codec_type") == "video"), None)
    if video is None:
        return None

    duration = payload.get("format", {}).get("duration") or video.get("duration")
    try:
        seconds = float(duration)
    except (TypeError, ValueError):
        seconds = 0.0
    return {
        "duration": round(seconds, 3),
        "width": int(video.get("width") or 0),
        "height": int(video.get("height") or 0),
        "ok": True,
    }


def _probe_with_ffmpeg(path: Path) -> dict:
    """Read the stream banner ffmpeg prints before it complains about no output.

    ffprobe is not shipped by the imageio-ffmpeg wheel, so on a machine with no
    system ffmpeg this is the only probe available. It is header-only and
    instant either way.
    """
    args = [ffmpeg_binary(), "-hide_banner", "-nostdin", "-i", str(path)]
    try:
        done = subprocess.run(args, capture_output=True, text=True,
                              errors="replace", timeout=120)
    except (OSError, subprocess.SubprocessError) as exc:
        return {"duration": 0.0, "width": 0, "height": 0, "ok": False,
                "error": f"{type(exc).__name__}: {exc}"}

    banner = done.stderr or ""
    size = _VIDEO_STREAM_RE.search(banner)
    if size is None:
        return {"duration": 0.0, "width": 0, "height": 0, "ok": False,
                "error": "no decodable video or image stream"}

    seconds = 0.0
    stamp = _DURATION_RE.search(banner)
    if stamp:
        seconds = int(stamp[1]) * 3600 + int(stamp[2]) * 60 + float(stamp[3])
    return {
        "duration": round(seconds, 3),
        "width": int(size[1]),
        "height": int(size[2]),
        "ok": True,
    }


def probe_media(path: str | Path) -> dict:
    """Measure a media file: {"duration", "width", "height", "ok"}.

    `duration` is 0.0 for stills. `ok` is False when the file cannot be read at
    all, which is how a corrupt download gets dropped from the edit instead of
    killing the whole encode.
    """
    target = Path(path)
    if not target.is_file():
        return {"duration": 0.0, "width": 0, "height": 0, "ok": False,
                "error": "file is missing"}
    return _probe_with_ffprobe(target) or _probe_with_ffmpeg(target)


# ---------------------------------------------------------------------------
# Inputs - step 3's audio and step 4's manifest, however they were handed over
# ---------------------------------------------------------------------------

def _resolve_audio(audio) -> tuple[Path, float]:
    """Accept step 3's result dict or a path. Returns (path, measured seconds)."""
    if isinstance(audio, dict):
        path = Path(audio["audio_path"])
        stated = float(audio.get("duration_seconds") or 0.0)
    else:
        path = Path(audio)
        stated = 0.0

    if not path.is_file():
        raise AssemblyError(f"narration audio not found: {path}")

    measured = 0.0
    try:  # mutagen is already a step-3 dependency and reads headers only
        from mutagen import File as MutagenFile

        info = MutagenFile(str(path))
        if info is not None and getattr(info, "info", None) is not None:
            measured = round(float(info.info.length), 3)
    except Exception:
        measured = 0.0

    if not measured:
        measured = probe_media(path).get("duration") or 0.0
    duration = measured or stated
    if duration <= 0:
        raise AssemblyError(f"could not measure the length of {path.name}")
    return path, round(float(duration), 3)


def _resolve_visuals(visuals) -> dict:
    """Accept step 4's manifest dict, its manifest.json, or the run directory."""
    if isinstance(visuals, dict):
        manifest = visuals
    else:
        path = Path(visuals)
        if path.is_dir():
            path = path / "manifest.json"
        if not path.is_file():
            raise AssemblyError(f"visuals manifest not found: {path}")
        try:
            manifest = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise AssemblyError(f"could not read {path}: {exc}") from exc

    for required in ("assets", "beats"):
        if required not in manifest:
            raise AssemblyError(
                f"{required!r} is missing from the visuals manifest - is this "
                f"a step 4 manifest.json?"
            )
    if not manifest["assets"]:
        raise NoFootageError("step 4's manifest lists no assets at all.")
    return manifest


def _asset_path(asset: dict, asset_dir: Path) -> Path:
    """Where an asset lives now.

    The manifest records an absolute path, but a run directory that has been
    moved (or written on another machine) still resolves through the relative
    name, so that is tried first.
    """
    relative = asset.get("relative_path")
    if relative:
        candidate = asset_dir / relative
        if candidate.is_file():
            return candidate
    return Path(asset.get("path", ""))


# ---------------------------------------------------------------------------
# The edit
# ---------------------------------------------------------------------------

@dataclass
class Shot:
    """One uninterrupted piece of footage on the finished timeline."""

    index: int
    beat_index: int
    asset_index: int
    path: Path
    kind: str
    start: float
    duration: float
    #: In-point inside the source clip. Always 0.0 for stills and looped clips.
    source_start: float = 0.0
    #: The clip is shorter than its slot, so it is played more than once.
    looped: bool = False
    #: This asset is already on the timeline somewhere earlier.
    reused: bool = False

    @property
    def end(self) -> float:
        return round(self.start + self.duration, 3)


@dataclass
class _Slot:
    """A stretch of narration that needs pictures over it."""

    beat_index: int
    start: float
    end: float
    assets: list[int] = dataclass_field(default_factory=list)

    @property
    def duration(self) -> float:
        return self.end - self.start


def _beat_slots(manifest: dict, usable: dict[int, dict], total: float,
                scale: float, warnings: list[str]) -> list[_Slot]:
    """Step 4's beats as timeline windows, with empty ones folded away.

    Beat boundaries come straight from the manifest - step 4 timed them against
    the measured .mp3, so they already line up with the narration. `scale`
    stretches them if the audio has since been re-rendered at a different
    length. The final beat is extended to cover VIDEO_TAIL_SECONDS.
    """
    slots: list[_Slot] = []
    for beat in manifest["beats"]:
        slots.append(_Slot(
            beat_index=int(beat["index"]),
            start=round(float(beat["start"]) * scale, 3),
            end=round(float(beat["end"]) * scale, 3),
            assets=[i for i in beat.get("asset_indexes", []) if i in usable],
        ))

    if not slots:  # assets but no beats: one slot for the lot
        slots = [_Slot(beat_index=0, start=0.0, end=total,
                       assets=sorted(usable))]

    slots[0].start = 0.0
    slots[-1].end = total
    for earlier, later in zip(slots, slots[1:]):
        later.start = earlier.end  # close rounding gaps between beats

    # A beat step 4 could not source anything for has no pictures of its own;
    # give its time to a neighbour rather than leaving a hole in the video.
    if any(slot.assets for slot in slots):
        merged: list[_Slot] = []
        for slot in slots:
            if slot.assets or not merged:
                merged.append(slot)
                continue
            warnings.append(
                f"beat {slot.beat_index} has no usable footage - its "
                f"{slot.duration:.1f}s was absorbed by beat {merged[-1].beat_index}"
            )
            merged[-1].end = slot.end
        # A leading empty beat cannot borrow from the left, so pull it right.
        while len(merged) > 1 and not merged[0].assets:
            warnings.append(
                f"beat {merged[0].beat_index} has no usable footage - its "
                f"{merged[0].duration:.1f}s was absorbed by beat "
                f"{merged[1].beat_index}"
            )
            merged[1].start = merged[0].start
            merged.pop(0)
        slots = merged
    return slots


def _shot_count(window: float, available: int) -> int:
    """How many cuts a window of narration should get.

    Aims for SHOT_TARGET_SECONDS a shot, then respects the floor and the
    ceiling. An asset may be used twice inside one beat if that is what the
    ceiling needs, but no further - past that the shot simply runs long.
    """
    target = max(0.5, config.SHOT_TARGET_SECONDS)
    count = max(1, int(round(window / target)))

    while count > 1 and window / count < config.SHOT_MIN_SECONDS:
        count -= 1
    if config.SHOT_MAX_SECONDS > 0:
        count = max(count, math.ceil(window / config.SHOT_MAX_SECONDS - 1e-6))
    return max(1, min(count, max(1, available * 2)))


def _source_in_point(clip_duration: float, slot: float) -> float:
    """Where to cut into a clip that is longer than the slot it has to fill.

    Stock footage often opens on a fade-up or a held frame, so a clip with room
    to spare starts a little way in.
    """
    spare = clip_duration - slot
    if spare <= 0.05 or clip_duration < slot * 2:
        return 0.0
    return round(min(spare, clip_duration * config.SHOT_SKIP_HEAD_RATIO), 3)


def _shows_the_same_thing(first: Shot, second: Shot) -> bool:
    """True if cutting from `first` to `second` would not change the picture.

    Happens when a beat is so short of footage that an asset has to be used
    twice running. A repeated *clip* is fine - it comes back at a different
    in-point, so it reads as a real cut - but a repeated still, or a clip that
    is being looped anyway, restarts on the frame it just left.
    """
    if first.asset_index != second.asset_index:
        return False
    if first.kind == "image":
        return True
    return (first.looped and second.looped) or (
        abs(first.source_start - second.source_start) < 0.05
    )


def _merge_identical_neighbours(shots: list[Shot]) -> list[Shot]:
    """Fold together consecutive shots that would look like one continuous take.

    A cut between two identical frames is not a cut, and for a still it is worse
    than none: the push-in snaps back to where it started. One longer shot moves
    smoothly through the whole window instead.
    """
    merged: list[Shot] = []
    for shot in shots:
        if merged and _shows_the_same_thing(merged[-1], shot):
            merged[-1].duration = round(merged[-1].duration + shot.duration, 3)
            continue
        shot.index = len(merged)
        merged.append(shot)
    return merged


def plan_timeline(manifest: dict, audio_duration: float,
                  probes: dict[int, dict]) -> tuple[list[Shot], list[str]]:
    """Lay step 4's assets out across the narration.

    Beat order and beat timings are taken as given; the only decisions made here
    are how many cuts each beat gets and which of its assets fill them.
    """
    warnings: list[str] = []
    assets = manifest["assets"]
    asset_dir = Path(manifest.get("asset_dir") or ".")

    usable = {i: a for i, a in enumerate(assets) if probes.get(i, {}).get("ok")}
    dropped = len(assets) - len(usable)
    if dropped:
        warnings.append(
            f"{dropped} of {len(assets)} sourced assets could not be read and "
            f"were left out of the edit"
        )
    if not usable:
        raise NoFootageError(
            "none of step 4's assets could be opened by ffmpeg - the run "
            "directory looks empty or corrupt."
        )

    total = round(audio_duration + max(0.0, config.VIDEO_TAIL_SECONDS), 3)
    stated = float(manifest.get("narration_duration") or 0.0)
    scale = 1.0
    if stated > 0 and abs(stated - audio_duration) > 0.25:
        scale = audio_duration / stated
        warnings.append(
            f"the manifest was built against a {stated:.1f}s narration but the "
            f"audio is {audio_duration:.1f}s - beat timings rescaled by {scale:.3f}"
        )

    slots = _beat_slots(manifest, usable, total, scale, warnings)
    used: set[int] = set()

    shots: list[Shot] = []
    for slot in slots:
        available = list(slot.assets)

        # A beat is filled from its own footage and nobody else's: step 4 chose
        # these clips for *this* sentence, and a neighbour's clip would be about
        # something else. _beat_slots has already folded away any beat with
        # nothing, and _shot_count never asks for more than two passes over what
        # is here, so cycling is always enough.
        wanted = _shot_count(slot.duration, len(available))
        chosen = [available[position % len(available)] for position in range(wanted)]

        share = slot.duration / len(chosen)
        if share > config.SHOT_MAX_SECONDS + 0.5:
            warnings.append(
                f"beat {slot.beat_index} holds one shot for {share:.1f}s - "
                f"step 4 sourced only {len(available)} usable asset(s) for it"
            )

        seen_here: dict[int, int] = {}
        for position, asset_index in enumerate(chosen):
            asset = assets[asset_index]
            start = round(slot.start + position * share, 3)
            duration = round(
                (slot.end - start) if position == len(chosen) - 1 else share, 3
            )
            repeat = seen_here.get(asset_index, 0)
            seen_here[asset_index] = repeat + 1

            kind = asset.get("kind", "video")
            clip_length = probes[asset_index].get("duration") or 0.0
            looped = False
            in_point = 0.0
            if kind == "video":
                if clip_length <= 0 or clip_length < duration + 0.05:
                    # Unknown or too short: let ffmpeg loop it and trim to the slot.
                    looped = True
                else:
                    in_point = _source_in_point(clip_length, duration)
                    # A second pass over the same clip cuts in somewhere else, so
                    # the repeat does not read as a glitch.
                    if repeat:
                        in_point = round(max(0.0, min(
                            clip_length - duration, in_point + repeat * duration
                        )), 3)

            shots.append(Shot(
                index=len(shots),
                beat_index=slot.beat_index,
                asset_index=asset_index,
                path=_asset_path(asset, asset_dir),
                kind=kind,
                start=start,
                duration=duration,
                source_start=in_point,
                looped=looped,
                reused=asset_index in used or repeat > 0,
            ))
            used.add(asset_index)

    shots = _merge_identical_neighbours(shots)
    if shots:  # absorb the rounding drift rather than ending a frame short
        shots[-1].duration = round(total - shots[-1].start, 3)

    spare = len(usable) - len(used)
    if spare:
        warnings.append(
            f"{spare} sourced asset(s) were not needed - step 4 found "
            f"{manifest.get('coverage_ratio', '?')}x more footage than the "
            f"narration runs for"
        )
    return shots, warnings


# ---------------------------------------------------------------------------
# Word timings
# ---------------------------------------------------------------------------

@dataclass
class Cue:
    """A burst of caption text: a few words, on screen together."""

    words: list[dict]
    start: float
    end: float

    @property
    def text(self) -> str:
        return " ".join(word["text"] for word in self.words)


def _clean_words(raw: list[dict]) -> list[dict]:
    """Drop the whitespace tokens aligners emit, and normalise the fields."""
    words = []
    for item in raw:
        text = (item.get("text") or item.get("word") or "").strip()
        if not text:
            continue
        try:
            start = float(item["start"])
            end = float(item["end"])
        except (KeyError, TypeError, ValueError):
            continue
        words.append({"text": text, "start": round(start, 3),
                      "end": round(max(start, end), 3)})
    return words


def _align_elevenlabs(audio_path: Path, script: str) -> tuple[list[dict], float]:
    """ElevenLabs forced alignment: the audio, plus the text it was made from.

    Forced alignment, not transcription - the words are the script's own, so
    there is nothing to mis-hear. One API call, no synthesis credits.
    """
    import requests

    api_key = config.require(*config.REQUIRED_TTS)["ELEVENLABS_API_KEY"]
    try:
        with open(audio_path, "rb") as handle:
            response = requests.post(
                ALIGNMENT_URL,
                headers={"xi-api-key": api_key},
                files={"file": (audio_path.name, handle, "audio/mpeg")},
                data={"text": script},
                timeout=ALIGNMENT_TIMEOUT,
            )
    except requests.exceptions.RequestException as exc:
        raise AssemblyError(
            f"could not reach the ElevenLabs alignment API: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    if response.status_code != 200:
        raise AssemblyError(
            f"ElevenLabs alignment failed [{response.status_code}]: "
            f"{(response.text or '')[:300]}"
        )

    payload = response.json()
    words = _clean_words(payload.get("words") or [])
    if not words:
        raise AssemblyError("ElevenLabs alignment returned no words")
    return words, float(payload.get("loss") or 0.0)


def _align_whisper(audio_path: Path) -> tuple[list[dict], float]:
    """faster-whisper locally, transcribing with word timestamps."""
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise AssemblyError(
            "faster-whisper is not installed (pip install faster-whisper)"
        ) from exc

    model = WhisperModel(config.CAPTION_WHISPER_MODEL, device="auto",
                         compute_type="int8")
    segments, _info = model.transcribe(str(audio_path), word_timestamps=True)
    words = _clean_words([
        {"text": word.word, "start": word.start, "end": word.end}
        for segment in segments for word in (segment.words or [])
    ])
    if not words:
        raise AssemblyError("whisper returned no word timestamps")
    return words, 0.0


def _syllables(word: str) -> int:
    """Rough syllable count - a better length proxy than characters alone."""
    letters = re.sub(r"[^a-z]", "", word.lower())
    if not letters:
        return 1
    count = len(re.findall(r"[aeiouy]+", letters))
    if letters.endswith("e") and count > 1 and not letters.endswith(("le", "ee")):
        count -= 1
    return max(1, count)


def _align_estimate(manifest: dict, audio_duration: float) -> tuple[list[dict], float]:
    """Spread each beat's words across that beat's window, by syllable weight.

    No network and no model. Step 4 already timed the beats against the measured
    .mp3, so the drift stays inside a beat instead of accumulating over the reel.
    """
    stated = float(manifest.get("narration_duration") or audio_duration)
    scale = audio_duration / stated if stated > 0 else 1.0

    words: list[dict] = []
    for beat in manifest["beats"]:
        tokens = [t for t in re.split(r"\s+", beat.get("text", "")) if t]
        if not tokens:
            continue
        start = float(beat["start"]) * scale
        end = float(beat["end"]) * scale
        weights = [_syllables(token) + (0.8 if _SENTENCE_END.search(token) else 0)
                   for token in tokens]
        total = sum(weights) or len(tokens)
        cursor = start
        for token, weight in zip(tokens, weights):
            span = (end - start) * weight / total
            words.append({"text": token, "start": round(cursor, 3),
                          "end": round(cursor + span, 3)})
            cursor += span
    if not words:
        raise AssemblyError("the manifest's beats carry no text to caption")
    return words, 0.0


def align_words(audio_path: Path, manifest: dict, audio_duration: float,
                mode: str) -> tuple[list[dict], str, float, list[str]]:
    """Word-level timings for the narration.

    Returns (words, backend, loss, warnings). Under "auto" the backends are
    tried best-first and a failure falls through to the next, so the captions
    are only ever as bad as the estimate - never absent.
    """
    warnings: list[str] = []
    script = (manifest.get("script") or "").strip()

    if mode == "auto":
        order = ["elevenlabs", "whisper", "estimate"]
        if not config.get_env("ELEVENLABS_API_KEY") or not script:
            order.remove("elevenlabs")
    else:
        order = [mode]
    if "estimate" not in order:
        order.append("estimate")

    backends = {
        "elevenlabs": lambda: _align_elevenlabs(audio_path, script),
        "whisper": lambda: _align_whisper(audio_path),
        "estimate": lambda: _align_estimate(manifest, audio_duration),
    }

    for backend in order:
        run = backends.get(backend)
        if run is None:
            raise AssemblyError(f"unknown alignment backend {backend!r}")
        if backend == "elevenlabs" and not script:
            warnings.append("caption alignment via elevenlabs skipped: the "
                            "manifest carries no script to align against")
            continue
        try:
            words, loss = run()
            return words, backend, loss, warnings
        except Exception as exc:
            warnings.append(f"caption alignment via {backend} failed: {exc}")

    raise AssemblyError(
        "every caption alignment backend failed: " + " | ".join(warnings)
    )


def group_cues(words: list[dict], limit: float) -> list[Cue]:
    """Break the word stream into short, readable bursts.

    A cue ends at the word limit, the character limit, a sentence ending or a
    pause - whichever comes first. Cues are then held until the next one starts,
    so short gaps do not blink the text off and on.
    """
    max_words = max(1, config.CAPTION_MAX_WORDS)
    max_chars = max(4, config.CAPTION_MAX_CHARS)

    cues: list[Cue] = []
    current: list[dict] = []

    def flush() -> None:
        if current:
            cues.append(Cue(list(current), current[0]["start"], current[-1]["end"]))
            current.clear()

    for word in words:
        if current:
            gap = word["start"] - current[-1]["end"]
            width = len(" ".join(w["text"] for w in current)) + 1 + len(word["text"])
            if (len(current) >= max_words or width > max_chars
                    or gap > CUE_SPLIT_GAP_SECONDS
                    or _SENTENCE_END.search(current[-1]["text"])):
                flush()
        current.append(word)
    flush()

    for index, cue in enumerate(cues):
        following = cues[index + 1].start if index + 1 < len(cues) else limit
        if following - cue.end <= CUE_HOLD_GAP_SECONDS:
            cue.end = following
        else:
            cue.end = cue.end + CUE_LINGER_SECONDS
        cue.end = max(cue.end, min(cue.start + config.CAPTION_MIN_CUE_SECONDS,
                                   following))
        cue.end = min(cue.end, following, limit)
    return [cue for cue in cues if cue.end > cue.start]


# ---------------------------------------------------------------------------
# Captions - ASS for the burn-in, SRT alongside it
# ---------------------------------------------------------------------------

def _font_family(path: Path) -> str | None:
    """The name libass will match on, read out of the font file itself."""
    try:
        from PIL import ImageFont

        family, style = ImageFont.truetype(str(path)).getname()
    except Exception:
        return None
    if not family:
        return None
    if style and style.lower() not in ("regular", "book", "normal"):
        return f"{family} {style}"
    return family


def resolve_font() -> tuple[str, Path | None, Path | None]:
    """Pick the caption face: (family name, file, fonts directory for libass).

    In order: CAPTION_FONT_FILE, then anything dropped into assets/fonts/, then
    CAPTION_FONT_NAME among the installed system fonts. The directory is only
    returned for project fonts - libass finds system fonts by name on its own,
    and pointing it at the system font directory costs seconds of scanning.
    """
    if config.CAPTION_FONT_FILE:
        chosen = Path(config.CAPTION_FONT_FILE).expanduser()
        if not chosen.is_file():
            raise AssemblyError(
                f"CAPTION_FONT_FILE points at {chosen}, which does not exist."
            )
        return (_font_family(chosen) or config.CAPTION_FONT_NAME,
                chosen, chosen.parent)

    wanted = config.CAPTION_FONT_NAME.lower()
    if config.FONTS_DIR.is_dir():
        local = sorted(p for p in config.FONTS_DIR.iterdir()
                       if p.suffix.lower() in _FONT_SUFFIXES)
        preferred = next(
            (p for p in local if (_font_family(p) or "").lower() == wanted), None
        )
        chosen = preferred or (local[0] if local else None)
        if chosen is not None:
            return (_font_family(chosen) or config.CAPTION_FONT_NAME,
                    chosen, config.FONTS_DIR)

    # Nothing project-local: name the font and let libass find it, but locate
    # the file too if we cheaply can - it is what measures the text below.
    name = config.CAPTION_FONT_NAME
    directories = [d for d in _SYSTEM_FONT_DIRS.get(sys.platform, ()) if d.is_dir()]
    for hint in _FONT_FILE_HINTS.get(wanted, ()):
        for directory in directories:
            if (directory / hint).is_file():
                return name, directory / hint, None
    for directory in directories:
        for candidate in sorted(directory.glob("*.ttf")):
            if (_font_family(candidate) or "").lower() == wanted:
                return name, candidate, None
    return name, None, None


def _ass_time(seconds: float) -> str:
    """H:MM:SS.cc - ASS's own timestamp format."""
    seconds = max(0.0, seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, rest = divmod(rest, 60)
    return f"{int(hours)}:{int(minutes):02d}:{rest:05.2f}"


def _srt_time(seconds: float) -> str:
    seconds = max(0.0, seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, rest = divmod(rest, 60)
    whole = int(rest)
    return (f"{int(hours):02d}:{int(minutes):02d}:{whole:02d},"
            f"{int(round((rest - whole) * 1000)):03d}")


def _ass_colour(value: str) -> str:
    """#RRGGBB -> ASS's &HAABBGGRR, opaque."""
    text = (value or "").strip().lstrip("#")
    if len(text) == 3:
        text = "".join(char * 2 for char in text)
    if not re.fullmatch(r"[0-9a-fA-F]{6}", text):
        raise AssemblyError(f"caption colour {value!r} is not a #RRGGBB hex value")
    return f"&H00{text[4:6]}{text[2:4]}{text[0:2]}".upper()


def _ass_inline_colour(value: str) -> str:
    """The same colour as an inline `\\c` override, for tinting one word."""
    return "{\\c&H" + _ass_colour(value)[4:] + "&}"


def _ass_escape(text: str) -> str:
    """Keep caption text from being read as ASS markup."""
    return (text.replace("\\", "").replace("{", "(").replace("}", ")")
                .replace("\n", " ").strip())


def _cue_font_size(text: str, font_file: Path | None) -> int:
    """Shrink a cue that would otherwise run past the frame edge.

    A long single word - "bioluminescence" - cannot be wrapped, so it is scaled
    down instead. Measured with the real font when its file was found, and with
    a conservative width-per-character otherwise.
    """
    base = config.CAPTION_FONT_SIZE
    # The outline grows the text on both sides and the shadow trails it on one,
    # so the glyphs get less than the bare margin-to-margin width. Leaving that
    # out lets a cue that measures as an exact fit wrap onto a second line.
    ink = 2 * config.CAPTION_OUTLINE_WIDTH + config.CAPTION_SHADOW_DEPTH
    usable = CAPTION_REFERENCE_WIDTH - 2 * config.CAPTION_MARGIN_H - ink
    if usable <= 0 or not text:
        return base

    width = 0.0
    if font_file is not None:
        try:
            from PIL import ImageFont

            face = ImageFont.truetype(str(font_file), base)
            ascent, descent = face.getmetrics()
            # libass asks FreeType for a *real dimension* size: the number it is
            # given is ascender-to-descender, not the em square Pillow measures
            # against. Without this correction a heavy face like Arial Black
            # measures ~40% wider than it renders, and every cue would shrink
            # itself away from a frame edge it was never going to touch.
            width = face.getlength(text) * base / max(1, ascent + descent)
        except Exception:
            width = 0.0
    if not width:  # no readable font file: assume a wide display face
        width = len(text) * base * 0.55

    if width <= usable:
        return base
    return max(24, int(base * usable / width))


def build_ass(cues: list[Cue], font_name: str, font_file: Path | None) -> str:
    """The subtitle file libass burns in.

    One dialogue line per cue in "plain" style; in "highlight" style one line
    per word, each showing the whole cue with the word being spoken tinted -
    which is what reads as the caption keeping up with the voice.
    """
    primary = _ass_colour(config.CAPTION_PRIMARY_COLOR)
    highlight = _ass_colour(config.CAPTION_HIGHLIGHT_COLOR)
    outline = _ass_colour(config.CAPTION_OUTLINE_COLOR)
    primary_inline = _ass_inline_colour(config.CAPTION_PRIMARY_COLOR)
    highlight_inline = _ass_inline_colour(config.CAPTION_HIGHLIGHT_COLOR)

    header = [
        "[Script Info]",
        "; Generated by pipeline/assemble.py - edit config.py, not this file.",
        "ScriptType: v4.00+",
        f"PlayResX: {CAPTION_REFERENCE_WIDTH}",
        f"PlayResY: {CAPTION_REFERENCE_HEIGHT}",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "YCbCr Matrix: TV.709",
        "",
        "[V4+ Styles]",
        "Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
        "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
        "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
        "MarginL, MarginR, MarginV, Encoding",
        (f"Style: Reel,{font_name},{config.CAPTION_FONT_SIZE},{primary},"
         f"{highlight},{outline},&H80000000,0,0,0,0,100,100,0,0,1,"
         f"{config.CAPTION_OUTLINE_WIDTH:g},{config.CAPTION_SHADOW_DEPTH:g},2,"
         f"{config.CAPTION_MARGIN_H},{config.CAPTION_MARGIN_H},"
         f"{config.CAPTION_MARGIN_V},1"),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text",
    ]

    frame = 1.0 / max(1, config.VIDEO_FPS)
    lines: list[str] = []
    for cue in cues:
        spoken = [_ass_escape(word["text"]) for word in cue.words]
        if config.CAPTION_UPPERCASE:
            spoken = [word.upper() for word in spoken]
        spoken = [word for word in spoken if word]
        if not spoken:
            continue

        joined = " ".join(spoken)
        size = _cue_font_size(joined, font_file)
        prefix = f"{{\\fs{size}}}" if size != config.CAPTION_FONT_SIZE else ""

        if config.CAPTION_STYLE != "highlight" or len(spoken) == 1:
            body = joined if config.CAPTION_STYLE != "highlight" else (
                f"{highlight_inline}{joined}{primary_inline}"
            )
            lines.append(f"Dialogue: 0,{_ass_time(cue.start)},"
                         f"{_ass_time(cue.end)},Reel,,0,0,0,,{prefix}{body}")
            continue

        # Segment boundaries: the cue opens on its first word, and each later
        # word takes over when it is spoken. Nudged apart so a fast word still
        # gets a frame of its own, and spread evenly if that overruns the cue.
        points = [cue.start] + [w["start"] for w in cue.words[1:]] + [cue.end]
        for i in range(1, len(points)):
            points[i] = max(points[i], points[i - 1] + frame)
        if points[-1] > cue.end + 1e-6:
            span = cue.end - cue.start
            points = [cue.start + span * i / len(spoken)
                      for i in range(len(spoken) + 1)]

        for position, _word in enumerate(spoken):
            tinted = " ".join(
                f"{highlight_inline}{token}{primary_inline}"
                if index == position else token
                for index, token in enumerate(spoken)
            )
            lines.append(
                f"Dialogue: 0,{_ass_time(points[position])},"
                f"{_ass_time(points[position + 1])},Reel,,0,0,0,,{prefix}{tinted}"
            )

    return "\n".join(header + lines) + "\n"


def build_srt(cues: list[Cue]) -> str:
    """The same cues as plain SRT - handy for checking sync outside a player."""
    blocks = []
    for number, cue in enumerate(cues, start=1):
        text = cue.text.upper() if config.CAPTION_UPPERCASE else cue.text
        blocks.append(
            f"{number}\n{_srt_time(cue.start)} --> {_srt_time(cue.end)}\n{text}\n"
        )
    return "\n".join(blocks)


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def _filter_path(path: Path) -> str:
    """Quote a path for use *inside* an ffmpeg filter argument.

    Filter arguments are parsed twice, so a Windows drive letter has to have its
    colon escaped or it reads as an option separator. Paths under the project go
    in relative instead, which sidesteps the problem entirely - ffmpeg is run
    with the project root as its working directory.
    """
    resolved = Path(path).resolve()
    try:
        text = resolved.relative_to(config.BASE_DIR).as_posix()
    except ValueError:
        text = resolved.as_posix().replace(":", r"\:")
    return "'" + text.replace("'", r"\'") + "'"


def _shot_inputs(shot: Shot) -> list[str]:
    """The `-ss`/`-t`/`-i` group that feeds one shot into the graph."""
    if shot.kind == "image":
        if config.IMAGE_ZOOM > 1.0:
            # zoompan turns a *single* input frame into the whole shot. Looping
            # the input as well would multiply the two and emit minutes of video
            # for a three-second slot.
            return ["-i", str(shot.path)]
        return ["-loop", "1", "-framerate", str(config.VIDEO_FPS),
                "-t", f"{shot.duration:.3f}", "-i", str(shot.path)]

    args: list[str] = []
    if shot.looped:
        args += ["-stream_loop", "-1"]
    elif shot.source_start > 0:
        args += ["-ss", f"{shot.source_start:.3f}"]
    return args + ["-t", f"{shot.duration:.3f}", "-i", str(shot.path)]


def _shot_chain(shot: Shot, label_in: str, label_out: str) -> str:
    """The filter chain that turns one source into a frame-shaped segment."""
    width, height, fps = config.VIDEO_WIDTH, config.VIDEO_HEIGHT, config.VIDEO_FPS

    if shot.kind == "image" and config.IMAGE_ZOOM > 1.0:
        frames = max(1, int(round(shot.duration * fps)))
        zoom = config.IMAGE_ZOOM
        # Oversample first: zoompan quantises its zoom to whole input pixels, so
        # a larger source makes the push-in smooth instead of steppy.
        big_w, big_h = int(width * 1.5) // 2 * 2, int(height * 1.5) // 2 * 2
        steps = [
            f"scale={big_w}:{big_h}:force_original_aspect_ratio=increase"
            f":flags=lanczos",
            f"crop={big_w}:{big_h}",
            (f"zoompan=z='min(1+{zoom - 1:.4f}*on/{frames},{zoom:.4f})'"
             f":d={frames}:s={width}x{height}:fps={fps}"
             f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"),
        ]
    else:
        steps = [
            f"scale={width}:{height}:force_original_aspect_ratio=increase"
            f":flags=lanczos",
            f"crop={width}:{height}",
            f"fps={fps}",
        ]

    steps += ["setsar=1", "format=yuv420p", "setpts=PTS-STARTPTS",
              f"trim=duration={shot.duration:.3f}", "setpts=PTS-STARTPTS"]

    if config.TRANSITION == "dip" and config.TRANSITION_SECONDS > 0:
        fade = min(config.TRANSITION_SECONDS, shot.duration / 3)
        steps.append(f"fade=t=in:st=0:d={fade:.3f}")
        steps.append(f"fade=t=out:st={shot.duration - fade:.3f}:d={fade:.3f}")

    return f"[{label_in}]" + ",".join(steps) + f"[{label_out}]"


def build_filter_graph(shots: list[Shot], audio_index: int, total: float,
                       narration: float, ass_path: Path | None,
                       fonts_dir: Path | None) -> tuple[str, str]:
    """The whole edit as one filter graph: shots, concat, captions, audio."""
    chains = [_shot_chain(shot, f"{i}:v", f"v{i}") for i, shot in enumerate(shots)]

    if len(shots) > 1:
        joined = "".join(f"[v{i}]" for i in range(len(shots)))
        chains.append(f"{joined}concat=n={len(shots)}:v=1:a=0[vcat]")
        stage = "vcat"
    else:
        stage = "v0"

    tail: list[str] = []
    if config.VIDEO_FADE_IN_SECONDS > 0:
        tail.append(f"fade=t=in:st=0:d={config.VIDEO_FADE_IN_SECONDS:g}")
    if config.VIDEO_FADE_OUT_SECONDS > 0:
        start = max(0.0, total - config.VIDEO_FADE_OUT_SECONDS)
        tail.append(f"fade=t=out:st={start:.3f}:d={config.VIDEO_FADE_OUT_SECONDS:g}")
    if ass_path is not None:
        burn = f"subtitles=filename={_filter_path(ass_path)}"
        if fonts_dir is not None:
            burn += f":fontsdir={_filter_path(fonts_dir)}"
        tail.append(burn)
    if tail:
        chains.append(f"[{stage}]" + ",".join(tail) + "[vout]")
        stage = "vout"

    # Order matters. loudnorm goes first: it is a dynamic normaliser, so a fade
    # applied ahead of it would be read as the narration going quiet and pulled
    # back up. It also runs internally at 192 kHz, hence the resample straight
    # after to land back on the output rate.
    audio = []
    if config.AUDIO_NORMALIZE:
        audio.append(f"loudnorm=I={config.AUDIO_LOUDNESS_TARGET:g}"
                     f":TP={config.AUDIO_LOUDNESS_PEAK:g}"
                     f":LRA={config.AUDIO_LOUDNESS_RANGE:g}")
    audio.append(f"aresample={config.AUDIO_SAMPLE_RATE}")
    if config.AUDIO_FADE_OUT_SECONDS > 0:
        start = max(0.0, narration - config.AUDIO_FADE_OUT_SECONDS)
        audio.append(f"afade=t=out:st={start:.3f}"
                     f":d={config.AUDIO_FADE_OUT_SECONDS:g}")
    # The picture outlives the narration by VIDEO_TAIL_SECONDS; pad the audio
    # with silence so the mux does not end the file early.
    audio.append("apad")
    chains.append(f"[{audio_index}:a]" + ",".join(audio) + "[aout]")

    return ";".join(chains), stage


def render(shots: list[Shot], audio_path: Path, output_path: Path, *,
           total: float, narration: float, ass_path: Path | None,
           fonts_dir: Path | None, work_dir: Path,
           overwrite: bool = True) -> list[str]:
    """Encode the reel. One ffmpeg pass: no intermediates, one generation loss."""
    if not shots:
        raise NoFootageError("there are no shots to render")

    inputs: list[str] = []
    for shot in shots:
        inputs += _shot_inputs(shot)
    audio_index = len(shots)
    inputs += ["-i", str(audio_path)]

    graph, stage = build_filter_graph(
        shots, audio_index, total, narration, ass_path, fonts_dir
    )

    args = [ffmpeg_binary(), "-hide_banner", "-nostdin",
            "-y" if overwrite else "-n", *inputs]
    if len(graph) > FILTER_GRAPH_INLINE_LIMIT:
        script = work_dir / "filtergraph.txt"
        script.write_text(graph, encoding="utf-8")
        args += ["-filter_complex_script", str(script)]
    else:
        args += ["-filter_complex", graph]

    args += [
        "-map", f"[{stage}]", "-map", "[aout]",
        "-t", f"{total:.3f}",
        "-c:v", config.VIDEO_CODEC,
        "-crf", str(config.VIDEO_CRF),
        "-preset", config.VIDEO_PRESET,
        "-profile:v", "high", "-level", "4.1",
        "-pix_fmt", config.VIDEO_PIXEL_FORMAT,
        "-r", str(config.VIDEO_FPS),
        "-c:a", config.AUDIO_CODEC,
        "-b:a", config.AUDIO_BITRATE,
        "-ar", str(config.AUDIO_SAMPLE_RATE),
        "-ac", "2",
        "-movflags", "+faststart",
        str(output_path),
    ]

    try:
        done = subprocess.run(
            args, capture_output=True, text=True, errors="replace",
            timeout=FFMPEG_TIMEOUT, cwd=str(config.BASE_DIR),
        )
    except subprocess.TimeoutExpired as exc:
        raise FFmpegError(
            f"ffmpeg did not finish within {FFMPEG_TIMEOUT}s and was killed"
        ) from exc
    except OSError as exc:
        raise FFmpegError(f"could not run ffmpeg: {exc}") from exc

    if done.returncode != 0:
        output_path.unlink(missing_ok=True)
        raise FFmpegError(
            f"ffmpeg exited {done.returncode}:\n"
            + (done.stderr or "")[-FFMPEG_ERROR_TAIL:]
        )

    return [
        line.strip() for line in (done.stderr or "").splitlines()
        if "Warning" in line or "Error" in line
    ][:5]


# ---------------------------------------------------------------------------
# Attribution carried over from step 4
# ---------------------------------------------------------------------------

def collect_attribution(manifest: dict, shots: list[Shot]) -> list[dict]:
    """Per-asset credits for the footage that actually made the cut.

    Step 4 sources more than the reel needs, so carrying everything it
    downloaded would credit work nobody ever sees. Screen time comes along too,
    which is what a licence audit would want to know.
    """
    assets = manifest["assets"]
    seconds: dict[int, float] = {}
    order: list[int] = []
    for shot in shots:
        if shot.asset_index not in seconds:
            order.append(shot.asset_index)
        seconds[shot.asset_index] = seconds.get(shot.asset_index, 0.0) + shot.duration

    credits = []
    for asset_index in order:
        asset = assets[asset_index]
        credits.append({
            "asset_index": asset_index,
            "beat_index": asset.get("beat_index"),
            "file": asset.get("relative_path") or Path(asset.get("path", "")).name,
            "kind": asset.get("kind"),
            "provider": asset.get("provider"),
            "provider_id": asset.get("provider_id"),
            "creator": asset.get("creator"),
            "creator_url": asset.get("creator_url"),
            "license": asset.get("license"),
            "source_url": asset.get("source_url"),
            "attribution": asset.get("attribution"),
            "attribution_required": asset.get("attribution_required", False),
            "screen_seconds": round(seconds[asset_index], 3),
        })
    return credits


def write_attribution_file(path: Path, manifest: dict, credits: list[dict]) -> None:
    """Human-readable credits next to the .mp4, for whoever posts it."""
    required = [c for c in credits if c.get("attribution_required")]
    lines = [
        f"Attribution for: {manifest.get('topic') or '(untitled)'}",
        f"Run: {manifest.get('run_id', '?')}",
        f"Assembled: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        f"{len(credits)} asset(s) appear in this video."
        + (f" {len(required)} require credit when posting." if required
           else " None require credit, but crediting is good manners."),
        "",
    ]
    for credit in credits:
        flag = "  [CREDIT REQUIRED]" if credit.get("attribution_required") else ""
        lines.append(
            f"- {credit['attribution'] or credit['file']}"
            f" ({credit['screen_seconds']}s on screen){flag}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assemble_video(audio, visuals, *, output_dir: str | Path | None = None,
                   run_id: str | None = None, captions: bool | None = None,
                   alignment: str | None = None, dry_run: bool = False,
                   overwrite: bool = True) -> dict:
    """Cut step 3's narration and step 4's footage into a finished reel.

    Args:
        audio: step 3's result dict, or a path to the narration .mp3.
        visuals: step 4's manifest dict, its manifest.json, or the run directory.
        output_dir: where to write. Defaults to output/<date>/<run id>/.
        run_id: overrides the manifest's run id, which names that directory.
        captions: burn in captions. Defaults to config.CAPTIONS_ENABLED.
        alignment: caption timing backend. Defaults to config.CAPTION_ALIGNMENT.
        dry_run: plan the edit and write the captions, but do not encode.
        overwrite: replace an existing .mp4 at the destination.

    Returns:
        The assembly manifest, also written to assembly.json in the output
        directory: the video's path and length, the shot-by-shot timeline, the
        caption files, and step 4's attribution for every asset used.

    Raises:
        AssemblyError (or a subclass) for anything that stops a reel being made.
    """
    warnings: list[str] = []
    manifest = _resolve_visuals(visuals)
    audio_path, narration = _resolve_audio(audio)

    use_captions = config.CAPTIONS_ENABLED if captions is None else captions
    backend_mode = (alignment or config.CAPTION_ALIGNMENT).strip().lower()
    identifier = run_id or manifest.get("run_id") or audio_path.stem

    destination = (Path(output_dir) if output_dir
                   else config.video_run_dir(identifier))
    destination.mkdir(parents=True, exist_ok=True)
    video_path = destination / config.VIDEO_FILENAME

    if video_path.exists() and not overwrite:
        raise AssemblyError(
            f"{video_path} already exists. Pass overwrite=True to replace it."
        )

    ffmpeg_binary()  # fail before any work if there is no encoder

    # --- plan the edit -----------------------------------------------------
    asset_dir = Path(manifest.get("asset_dir") or ".")
    probes = {
        index: probe_media(_asset_path(asset, asset_dir))
        for index, asset in enumerate(manifest["assets"])
    }
    shots, plan_warnings = plan_timeline(manifest, narration, probes)
    warnings += plan_warnings
    total = round(shots[-1].end, 3)

    # --- captions ----------------------------------------------------------
    caption_report = {
        "enabled": bool(use_captions), "backend": None, "cues": 0, "words": 0,
        "loss": None, "font": None, "style": config.CAPTION_STYLE,
        "ass_path": None, "srt_path": None, "alignment_path": None,
    }
    ass_path: Path | None = None
    fonts_dir: Path | None = None

    if use_captions:
        words, backend, loss, align_warnings = align_words(
            audio_path, manifest, narration, backend_mode
        )
        warnings += align_warnings
        cues = group_cues(words, total)
        font_name, font_file, fonts_dir = resolve_font()
        if font_file is None:
            warnings.append(
                f"no font file found for {font_name!r} - libass will substitute "
                f"if it cannot resolve the name either. Drop a .ttf into "
                f"assets/fonts/ to pin the look."
            )

        ass_path = destination / CAPTIONS_ASS_NAME
        ass_path.write_text(build_ass(cues, font_name, font_file), encoding="utf-8")
        srt_path = destination / CAPTIONS_SRT_NAME
        srt_path.write_text(build_srt(cues), encoding="utf-8")
        alignment_path = destination / ALIGNMENT_NAME
        alignment_path.write_text(
            json.dumps({"backend": backend, "loss": loss, "words": words},
                       indent=2, ensure_ascii=False), encoding="utf-8"
        )
        caption_report.update({
            "backend": backend, "cues": len(cues), "words": len(words),
            "loss": round(loss, 4) if loss else loss, "font": font_name,
            "ass_path": str(ass_path.resolve()),
            "srt_path": str(srt_path.resolve()),
            "alignment_path": str(alignment_path.resolve()),
        })

    # --- render ------------------------------------------------------------
    if dry_run:
        rendered = {"duration": total, "ok": False}
    else:
        warnings += render(
            shots, audio_path, video_path,
            total=total, narration=narration,
            ass_path=ass_path, fonts_dir=fonts_dir,
            work_dir=destination, overwrite=overwrite,
        )
        rendered = probe_media(video_path)
        if not rendered.get("ok"):
            raise FFmpegError(
                f"ffmpeg reported success but {video_path.name} cannot be read back"
            )
        if abs(rendered["duration"] - total) > 0.5:
            warnings.append(
                f"the rendered video is {rendered['duration']:.2f}s against a "
                f"planned {total:.2f}s"
            )

    # --- carry step 4's attribution forward --------------------------------
    credits = collect_attribution(manifest, shots)
    attribution_path = destination / ATTRIBUTION_NAME
    write_attribution_file(attribution_path, manifest, credits)

    assembly = {
        "assembly_version": ASSEMBLY_VERSION,
        "run_id": identifier,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "topic": manifest.get("topic"),
        "script": manifest.get("script"),
        "dry_run": dry_run,
        "video_path": str(video_path.resolve()),
        "output_dir": str(destination.resolve()),
        "duration_seconds": round(float(rendered.get("duration") or total), 3),
        "frame": {
            "width": config.VIDEO_WIDTH,
            "height": config.VIDEO_HEIGHT,
            "fps": config.VIDEO_FPS,
            "aspect": "9:16",
        },
        "audio_path": str(audio_path.resolve()),
        "narration_duration": narration,
        "tail_seconds": round(total - narration, 3),
        "visuals_manifest": manifest.get("manifest_path"),
        "asset_dir": manifest.get("asset_dir"),
        "captions": caption_report,
        "timeline": [
            {
                "index": shot.index,
                "beat_index": shot.beat_index,
                "asset_index": shot.asset_index,
                "file": shot.path.name,
                "kind": shot.kind,
                "start": shot.start,
                "duration": shot.duration,
                "source_start": shot.source_start,
                "looped": shot.looped,
                "reused": shot.reused,
            }
            for shot in shots
        ],
        "attributions": credits,
        "attribution_path": str(attribution_path.resolve()),
        "warnings": list(dict.fromkeys(warnings)),
        "stats": {
            "shots": len(shots),
            "beats": len({shot.beat_index for shot in shots}),
            "assets_used": len(credits),
            "assets_sourced": len(manifest["assets"]),
            "looped_shots": sum(1 for shot in shots if shot.looped),
            "reused_shots": sum(1 for shot in shots if shot.reused),
            "mean_shot_seconds": round(total / len(shots), 2) if shots else 0.0,
            "file_bytes": video_path.stat().st_size if video_path.is_file() else 0,
        },
    }

    (destination / ASSEMBLY_NAME).write_text(
        json.dumps(assembly, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return assembly


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_result(assembly: dict) -> None:
    rule = "-" * 68
    stats, captions = assembly["stats"], assembly["captions"]
    print(rule)
    print(f"RUN     : {assembly['run_id']}")
    print(f"TOPIC   : {assembly.get('topic') or '(none)'}")
    print(f"VIDEO   : {assembly['video_path']}"
          + ("  (dry run - not encoded)" if assembly["dry_run"] else ""))
    print(f"FRAME   : {assembly['frame']['width']}x{assembly['frame']['height']}"
          f" @ {assembly['frame']['fps']}fps ({assembly['frame']['aspect']})")
    print(f"LENGTH  : {assembly['duration_seconds']}s "
          f"({assembly['narration_duration']}s narration "
          f"+ {assembly['tail_seconds']}s tail)")
    print(f"SHOTS   : {stats['shots']} across {stats['beats']} beats, "
          f"~{stats['mean_shot_seconds']}s each "
          f"({stats['assets_used']}/{stats['assets_sourced']} assets used)")
    if captions["enabled"]:
        print(f"CAPTIONS: {captions['cues']} cues from {captions['words']} words "
              f"via {captions['backend']}"
              + (f" (loss {captions['loss']})" if captions["loss"] else "")
              + f", {captions['font']} {captions['style']}")
    else:
        print("CAPTIONS: off")
    if stats["file_bytes"]:
        print(f"SIZE    : {stats['file_bytes'] / 1024 / 1024:.1f} MB")
    print(f"CREDITS : {assembly['attribution_path']}")

    print(rule)
    print("TIMELINE  (L looped, R clip reused)")
    for shot in assembly["timeline"]:
        flags = "".join(char for char, on in (
            ("L", shot["looped"]), ("R", shot["reused"])
        ) if on)
        print(f"  {shot['start']:>6.2f}s +{shot['duration']:>5.2f}s  "
              f"beat {shot['beat_index']:<2} {shot['kind']:<5} "
              f"{flags:<3} {shot['file'][:44]}")

    if assembly["warnings"]:
        print(rule)
        print("WARNINGS")
        for warning in assembly["warnings"]:
            print(f"  - {warning}")
    print(rule)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Assemble a 9:16 reel from step 3's audio and step 4's visuals."
    )
    parser.add_argument("-a", "--audio", type=Path, required=True,
                        help="narration .mp3 from step 3")
    parser.add_argument("-v", "--visuals", type=Path, required=True,
                        help="step 4 run directory, or its manifest.json")
    parser.add_argument("-o", "--output-dir", type=Path, default=None,
                        help="output directory (default: output/<date>/<run id>/)")
    parser.add_argument("--run-id", default=None,
                        help="override the run id taken from the manifest")
    parser.add_argument("--alignment", default=None,
                        choices=("auto", "elevenlabs", "whisper", "estimate"),
                        help=f"caption timing source "
                             f"(default: {config.CAPTION_ALIGNMENT})")
    parser.add_argument("--no-captions", action="store_true",
                        help="render the video without burned-in captions")
    parser.add_argument("--dry-run", action="store_true",
                        help="plan the edit and write the captions, do not encode")
    parser.add_argument("--json", action="store_true",
                        help="print the assembly manifest as JSON")
    args = parser.parse_args(argv)

    try:
        assembly = assemble_video(
            args.audio, args.visuals,
            output_dir=args.output_dir,
            run_id=args.run_id,
            captions=False if args.no_captions else None,
            alignment=args.alignment,
            dry_run=args.dry_run,
        )
    except (AssemblyError, config.MissingConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    if args.json:
        print(json.dumps(assembly, indent=2, ensure_ascii=False))
    else:
        _print_result(assembly)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
