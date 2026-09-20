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
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field as dataclass_field
from datetime import datetime, timezone
from pathlib import Path

# Allow both `python pipeline/assemble.py` and `python -m pipeline.assemble`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from pipeline import look  # noqa: E402
from pipeline import pacing  # noqa: E402

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

#: What makes a caption word a payoff rather than a connective. Numbers are
#: the whole of it, and they are almost never digits: step 2's prompt says to
#: "write numbers and symbols the way they should be said aloud", so the
#: narration says "ninety-five percent" and the captions carry the words.
#: Measured over every script in output/scripts/ - twelve reels, 39 digit
#: characters in total, against "quadrillion", "ninety-five", "thousand",
#: "eleven" and the rest spelled out. A digit-only test would fire on nothing.
#:
#: Deliberately a second, smaller copy of the vocabulary in script_gen.py
#: rather than an import of it. That one folds number words onto their digits
#: for the structure pass's restatement detector, where the folding is load
#: bearing and was a real bug when it was missing; this one decides what colour
#: a word is drawn in. Sharing them would mean a retune of the structure pass
#: silently restyling the captions.
_PAYOFF_NUMBER_WORDS = frozenset("""
    zero one two three four five six seven eight nine ten
    eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen
    twenty thirty forty fifty sixty seventy eighty ninety
    hundred thousand million billion trillion quadrillion quintillion
    percent
""".split())

_PAYOFF_DIGIT_RE = re.compile(r"\d")

#: Punctuation a word may be wearing when it reaches the caption. Stripped
#: before the payoff test, so "quadrillion." is still a payoff.
_PAYOFF_STRIP = "\"'“”‘’.,!?;:()[]"

#: A cut may never leave a shot shorter than this, whatever snapping wants.
#: Well under the pacing curve's fastest opening, so it only ever bites on a
#: shot that was already too short to see.
_SNAP_MIN_SHOT_SECONDS = 0.5

#: A silence has to be at least this long to count as somewhere to cut. Below
#: it, what looks like a gap between two words is the stop consonant inside one
#: phrase, and cutting there is no better than cutting mid-word.
_SNAP_MIN_GAP_SECONDS = 0.06

#: How close to a beat a cut has to be to count as on it, when the finished
#: timeline is measured back against the grid. One frame at 30fps, rounded up:
#: the plan stores boundaries to the millisecond and the encode lands them on
#: whole frames, so exact equality would report a grid nothing sits on.
_BEAT_TOLERANCE_SECONDS = 0.035

#: How far inside a spoken word a beat has to fall before it is refused as a
#: landing. A beat on the attack of a word is the fallback `_snap_boundaries`
#: already reaches for by choice; a beat in the middle of one is the mid-word
#: cut that snapping exists to prevent, and the music does not get to override
#: that.
_BEAT_WORD_TOLERANCE = 0.04

#: How much bigger than the output frame a moving shot is scaled before
#: zoompan sees it. zoompan quantises its zoom to whole input pixels, so the
#: move is only as smooth as the picture it is cropping out of - at 1:1 a slow
#: push advances in visible steps. Not a config knob: it trades encode time for
#: smoothness on a curve with nothing interesting on it, and 1.5 is past the
#: knee. It is also why a moving shot never carries a `crop` to the output
#: size - zoompan's own `s=` does that at the end of the chain.
MOTION_OVERSAMPLE = 1.5

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

#: The card-only subtitle file the cover is drawn through. Kept rather than
#: written to a temp file: when a cover looks wrong, this is the thing to look at.
COVER_ASS_NAME = "cover.ass"

#: Where inside the chosen shot the cover frame is taken from, as a fraction of
#: its length. Mechanical rather than a matter of taste: far enough past the cut
#: that the picture has settled, far enough before the next one that the subject
#: has not already left the frame.
_COVER_SHOT_POSITION = 0.45

#: ...and never closer than this to either end of the shot, whatever the
#: fraction works out to on a very short one.
_COVER_EDGE_SECONDS = 0.25

#: Bumped to 2 when the cover frame was added - steps 6 and 9 read this file,
#: and version 1 has no `cover` in it. Bumped to 3 for the beat grid and the
#: caption pop: `music` gained `bpm`/`beat_offset`, `stats` gained
#: `cuts_on_beat` and `captions` gained `pop`/`payoff_words`. Additive, so
#: nothing downstream had to change - but step 9 correlates on these, and a
#: version 2 file having none of them is the difference between "zero" and
#: "not measured".
#: Bumped to 4 for step 5b: the file gained a top-level `look` with the grade
#: the rendered reel was given, and `cover` gained `look`/`look_candidates`
#: with what the model made of the frames it chose between. Additive again,
#: and the same distinction again - a version 3 file is not a reel that was
#: looked at and found clean, it is a reel from before anything looked. Step 9
#: draws that line with `look_graded` rather than by reading this number, so
#: the two stay independent: switching the look off has to be visible on a
#: current run, not only on an old one.
#: Bumped to 5 for the same distinction one level down: each entry in
#: `look.frames` gained `shot` and `beat`, naming the clip that verdict is
#: about. A version 4 file's verdicts are timestamps, and a timestamp only
#: names a shot if you go back to `timeline` and `timeline` agrees with the
#: render - which near a cut it does not, because the encode rounds every
#: shot to whole frames and the error accumulates. So a reader that re-derives
#: the shot from an old file will sometimes get the wrong one, and cannot tell
#: from the file that it has. This number is how it can.
ASSEMBLY_VERSION = 5


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

def probe_audio(path: str | Path) -> float:
    """Length of an audio-only file in seconds, or 0.0 if it cannot be read.

    probe_media is no use here: it looks for a decodable video or image stream
    and calls a file without one broken, which is exactly what a .mp3 is.
    mutagen reads headers for every format step 3 and the music bed use, and
    ffprobe's container duration covers whatever it does not.
    """
    path = Path(path)
    try:
        from mutagen import File as MutagenFile

        info = MutagenFile(str(path))
        if info is not None and getattr(info, "info", None) is not None:
            length = float(info.info.length)
            if length > 0:
                return round(length, 3)
    except Exception:
        pass

    exe = ffprobe_binary()
    if not exe:
        return 0.0
    try:
        done = subprocess.run(
            [exe, "-v", "error", "-show_entries", "format=duration",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=60,
        )
        seconds = float(json.loads(done.stdout or "{}")
                        .get("format", {}).get("duration") or 0.0)
    except (OSError, ValueError, TypeError, subprocess.SubprocessError):
        return 0.0
    return round(seconds, 3) if seconds > 0 else 0.0


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

    duration = probe_audio(path) or stated
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
    #: How the frame moves across this shot: "push", "pull", or "" for static.
    #: Decided by `_plan_motion` once the timeline is settled, never at render
    #: time - the direction alternates along the shot list, and merging two
    #: neighbours changes what that list is.
    motion: str = ""
    #: How far it travels, as a zoom factor. 1.0 is no movement.
    motion_zoom: float = 1.0

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
        slots = _absorb_empty_slots(slots, warnings)
    return slots


def _absorb_empty_slots(slots: list[_Slot], warnings: list[str]) -> list[_Slot]:
    """Fold beats with no footage into their neighbours, splitting the cost.

    A run of empty beats is shared between the beat before it and the beat
    after it, meeting in the middle, rather than being handed wholesale to
    whichever one happens to come first.

    This used to absorb strictly leftwards, which is fine for one empty beat
    and compounds badly for several: on the deep-ocean run, beats 5, 6 and 7
    all failed together and all 12.7s of them landed on beat 4, so two clips
    covered 18 of the reel's 45 seconds while beat 8 - which had footage, and
    sat directly against beat 7 - was untouched. Splitting the run does not
    make the pictures right (nothing here can; the fix for that is upstream in
    step 4) but it stops one shot holding for a third of the reel, which is
    the part a viewer actually feels.

    A run with no beat on one side goes entirely to the other: a leading run
    has nothing to its left to borrow from, and a trailing one nothing to its
    right.
    """
    merged: list[_Slot] = []
    index = 0
    while index < len(slots):
        if slots[index].assets:
            merged.append(slots[index])
            index += 1
            continue

        run_end = index
        while run_end < len(slots) and not slots[run_end].assets:
            run_end += 1
        run = slots[index:run_end]
        left = merged[-1] if merged else None
        right = slots[run_end] if run_end < len(slots) else None

        if left is not None and right is not None:
            boundary = round((run[0].start + run[-1].end) / 2, 3)
            left.end, right.start = boundary, boundary
            took = f"beats {left.beat_index} and {right.beat_index}"
        elif left is not None:      # trailing run: nothing to its right
            left.end = run[-1].end
            took = f"beat {left.beat_index}"
        elif right is not None:     # leading run: nothing to its left
            right.start = run[0].start
            took = f"beat {right.beat_index}"
        else:
            # Every beat is empty. The caller only gets here with at least one
            # slot holding footage, so this cannot happen - and if it ever
            # does, a reel with no pictures at all is not something to paper
            # over with a merge.
            break

        for slot in run:
            warnings.append(
                f"beat {slot.beat_index} has no usable footage - its "
                f"{slot.duration:.1f}s was absorbed by {took}"
            )
        index = run_end
    return merged


# The pacing curve itself lives in `pipeline.pacing`. Step 4 has to ask the same
# question this module answers - how many shots a beat becomes - before it can
# decide how many clips to source for that beat, and while the two steps
# answered it with different arithmetic a long beat was cut into more shots
# than it had footage for. The names are kept so the rest of this module reads
# as it always did.
shot_target_at = pacing.shot_target_at
_curve_shots_in = pacing.curve_shots_in
_shot_count = pacing.shot_count


def _shot_spans(start: float, end: float, count: int,
                total: float) -> list[tuple[float, float]]:
    """Divide [start, end] into `count` shots whose lengths follow the curve.

    Dividing the window evenly would put the right *number* of cuts in the
    opening beat and still space them identically, which is not what the curve
    asked for. Each shot is instead laid down at the length the curve wants
    where it begins, and the run is then rescaled to land exactly on `end` -
    so a beat that straddles the ramp opens short and finishes at the steady
    rate, rather than averaging the two.
    """
    span = end - start
    if count <= 1 or span <= 0:
        return [(start, end)]

    lengths: list[float] = []
    cursor = start
    for _ in range(count):
        length = max(0.3, shot_target_at(cursor, total))
        lengths.append(length)
        cursor += length

    scale = span / sum(lengths)
    spans: list[tuple[float, float]] = []
    cursor = start
    for length in lengths:
        following = cursor + length * scale
        spans.append((round(cursor, 3), round(following, 3)))
        cursor = following
    # The last shot takes the rounding, so the beat ends exactly where it must.
    spans[-1] = (spans[-1][0], round(end, 3))
    return spans


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


def _plan_motion(shots: list[Shot], mode: str | None = None) -> None:
    """Decide how the frame moves across each shot, in place.

    Runs after the timeline is final - after snapping and after identical
    neighbours have been merged - because the direction alternates along the
    shot list, and both of those change what that list is. A merge that folded
    two shots into one would otherwise leave a gap in the alternation.

    Three rules, in order of who wins:

    * the opening shot always pushes in, and harder than the rest. It is on
      screen for the second and a half in which the viewer decides, and it is
      the frame the cover is drawn from;
    * every other clip moves by SHOT_MOTION_ZOOM, pushing and pulling in turn
      under `alternate` so that a run of shots does not read as one move on
      repeat;
    * stills keep IMAGE_ZOOM and always push. They are a separate knob because
      they are a separate problem - a still has no motion of its own at all -
      and turning clip motion off has never been a reason to freeze them.

    Stills still take their turn in the alternation count rather than being
    skipped over, so the clips either side of one carry on in the order they
    would have had. A still simply does not honour the "pull" half of it.
    """
    # `mode` overrides the setting for one reel, which is what lets step 9 run
    # a motion trial: the column was answered "on" by every reel ever made, so
    # nothing in the ledger could say whether the move is worth the crop.
    mode = (mode if mode is not None else config.SHOT_MOTION or "off")
    mode = str(mode).strip().lower()
    clips_move = mode not in ("", "off", "none") and config.SHOT_MOTION_ZOOM > 1.0

    for shot in shots:
        if shot.index == 0 and clips_move:
            shot.motion, shot.motion_zoom = "push", config.SHOT_MOTION_HOOK_ZOOM
        elif shot.kind == "image":
            shot.motion = "push" if config.IMAGE_ZOOM > 1.0 else ""
            shot.motion_zoom = config.IMAGE_ZOOM
        elif clips_move:
            pull = mode == "alternate" and shot.index % 2 == 1
            shot.motion = "pull" if pull else "push"
            shot.motion_zoom = config.SHOT_MOTION_ZOOM
        else:
            shot.motion, shot.motion_zoom = "", 1.0

        if shot.motion_zoom <= 1.0:
            shot.motion, shot.motion_zoom = "", 1.0


def _word_cut_points(words: list[dict]) -> tuple[list[float], list[float]]:
    """Where a cut can land without falling through the middle of a word.

    Returns (gaps, starts): the midpoints of the silences between consecutive
    words, and every word's start. Gaps are the better landing - a cut in a
    pause is inaudible - and word starts are the fallback for continuous
    speech, where a cut on the attack of a syllable still reads as deliberate.
    """
    gaps: list[float] = []
    starts: list[float] = []
    for word in words:
        try:
            starts.append(float(word["start"]))
        except (KeyError, TypeError, ValueError):
            continue
    for earlier, later in zip(words, words[1:]):
        try:
            end, start = float(earlier["end"]), float(later["start"])
        except (KeyError, TypeError, ValueError):
            continue
        if start - end >= _SNAP_MIN_GAP_SECONDS:
            gaps.append((end + start) / 2)
    return sorted(gaps), sorted(starts)


def _nearest_within(points: list[float], target: float,
                    lower: float, upper: float) -> float | None:
    """The point closest to `target` that lies inside [lower, upper]."""
    best: float | None = None
    best_distance = float("inf")
    for point in points:
        if point < lower or point > upper:
            continue
        distance = abs(point - target)
        if distance < best_distance:
            best, best_distance = point, distance
    return best


def _spoken_spans(words: list[dict]) -> list[tuple[float, float]]:
    """When the voice is actually making a sound, as (start, end) pairs."""
    spans = []
    for word in words:
        try:
            spans.append((float(word["start"]), float(word["end"])))
        except (KeyError, TypeError, ValueError):
            continue
    return spans


def _snap_to_beats(plan: list[dict], beats: list[float],
                   words: list[dict]) -> list[int]:
    """Put the opening cuts on the music's beat grid.

    Returns the boundary indices it pinned. `_snap_boundaries` is given those
    to leave alone: run afterwards without them, the word pass would pull every
    cut straight back off the grid and whichever pass ran last would be the
    only one that had done anything.

    Only the opening is governed. MUSIC_FADE_IN_SECONDS has just finished, the
    narration has not built up and the cutting is at its fastest, so the bed is
    as audible as it will ever be; past SHOT_BEAT_WINDOW_SECONDS it has settled
    under the voice and cutting to a tempo nobody is listening for buys nothing.

    A beat that falls inside a spoken word is refused. A cut in the middle of a
    word is the defect `_snap_boundaries` exists to prevent, and the music does
    not get to override that - the boundary falls through to the word pass
    instead, which is the same place it would have gone with no music at all.
    """
    if len(plan) < 2 or not beats:
        return []

    limit = max(0.0, config.SHOT_BEAT_MAX_SECONDS)
    window = max(0.0, config.SHOT_BEAT_WINDOW_SECONDS)
    if limit <= 0 or window <= 0:
        return []

    spans = _spoken_spans(words)
    clear = [
        beat for beat in beats
        if not any(start + _BEAT_WORD_TOLERANCE < beat < end - _BEAT_WORD_TOLERANCE
                   for start, end in spans)
    ]
    if not clear:
        return []

    pinned: list[int] = []
    for index in range(1, len(plan)):
        boundary = plan[index]["start"]
        if boundary > window:
            break
        lower = max(plan[index - 1]["start"] + _SNAP_MIN_SHOT_SECONDS,
                    boundary - limit)
        upper = min(plan[index]["end"] - _SNAP_MIN_SHOT_SECONDS,
                    boundary + limit)
        if upper <= lower:
            continue

        landing = _nearest_within(clear, boundary, lower, upper)
        if landing is None:
            continue
        plan[index - 1]["end"] = plan[index]["start"] = round(landing, 3)
        pinned.append(index)
    return pinned


def _snap_boundaries(plan: list[dict], words: list[dict],
                     skip: list[int] | tuple[int, ...] = ()) -> int:
    """Nudge each cut onto the nearest word boundary. Returns how many moved.

    Step 5 already knows when every word is spoken, because it aligns the
    narration to write the captions. Without this, a cut lands wherever
    dividing a beat window by a shot count happened to put it - which is,
    reliably, halfway through a word. The picture changes mid-syllable and the
    edit reads as something that went wrong rather than something that was
    done.

    Only interior boundaries move: the reel still opens at 0 and still ends
    where the narration does. Each cut is constrained to leave both of the
    shots it separates at least _SNAP_MIN_SHOT_SECONDS long, so a nudge can
    never collapse a shot, and boundaries are walked in order so a moved cut
    becomes the floor for the next one.

    `skip` is the boundaries `_snap_to_beats` has already placed on the music.
    They are left exactly where they are - moving one onto a nearer pause is
    the same as taking it off the beat.
    """
    if len(plan) < 2 or not words:
        return 0

    gaps, starts = _word_cut_points(words)
    if not gaps and not starts:
        return 0

    limit = max(0.0, config.SHOT_SNAP_MAX_SECONDS)
    if limit <= 0:
        return 0

    pinned = set(skip)
    moved = 0
    for index in range(1, len(plan)):
        if index in pinned:
            continue            # the music has this one; see _snap_to_beats
        boundary = plan[index]["start"]
        lower = max(plan[index - 1]["start"] + _SNAP_MIN_SHOT_SECONDS,
                    boundary - limit)
        upper = min(plan[index]["end"] - _SNAP_MIN_SHOT_SECONDS,
                    boundary + limit)
        if upper <= lower:
            continue

        # A pause first; the attack of a word only if there is no pause to be
        # had within reach.
        landing = (_nearest_within(gaps, boundary, lower, upper)
                   or _nearest_within(starts, boundary, lower, upper))
        if landing is None or abs(landing - boundary) < 0.005:
            continue

        plan[index - 1]["end"] = plan[index]["start"] = round(landing, 3)
        moved += 1
    return moved


def plan_timeline(manifest: dict, audio_duration: float,
                  probes: dict[int, dict],
                  words: list[dict] | None = None,
                  beats: list[float] | None = None,
                  motion: str | None = None,
                  ) -> tuple[list[Shot], list[str]]:
    """Lay step 4's assets out across the narration.

    Beat order and beat timings are taken as given; the decisions made here are
    how many cuts each beat gets, which of its assets fill them, and - once
    `words` is on hand - exactly where those cuts land.

    It runs in three passes, and the order matters. Which asset covers which
    stretch is settled first; the cuts between them are then nudged onto word
    boundaries; and only after that is each shot asked how it should be played,
    because whether a clip has to loop depends on how long its slot ended up
    being, and snapping changes that by up to SHOT_SNAP_MAX_SECONDS either way.

    Args:
        words: aligned narration, for snapping cuts to word boundaries. Without
            it the cuts fall where the arithmetic put them, which is what
            happens when captions are off and SHOT_SNAP_TO_WORDS is not set.
        beats: the music bed's beat grid, from `beat_grid`, for putting the
            opening cuts on the tempo. Empty or absent - a reel with no music,
            or a track with no BPM written beside it - and the words take every
            cut, which is what every reel before SHOT_SNAP_TO_BEAT did.
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

    # --- pass 1: which asset covers which stretch --------------------------
    plan: list[dict] = []
    for slot in slots:
        available = list(slot.assets)

        # A beat is filled from its own footage and nobody else's: step 4 chose
        # these clips for *this* sentence, and a neighbour's clip would be about
        # something else. _beat_slots has already folded away any beat with
        # nothing, and _shot_count never asks for more than two passes over what
        # is here, so cycling is always enough.
        wanted = _shot_count(slot.duration, len(available),
                             position=slot.start, total=total)
        chosen = [available[position % len(available)] for position in range(wanted)]

        # Step 4 now sources a clip per shot, so this is rare and worth saying
        # out loud when it happens: the run is about to put footage back on
        # screen that the same beat showed a few seconds ago, and until now the
        # only trace of that was a look grade after the encode. What is left of
        # it is beats that inherited seconds from a neighbour with no usable
        # footage, which step 4 cannot see coming - 2 of the 151 beats on disk.
        if wanted > len(available):
            short = wanted - len(available)
            warnings.append(
                f"beat {slot.beat_index} was cut into {wanted} shots from "
                f"{len(available)} clip(s) - "
                f"{'one of them re-shows' if short == 1 else f'{short} of them re-show'} "
                f"footage from earlier in the same beat"
            )

        spans = _shot_spans(slot.start, slot.end, len(chosen), total)
        longest = max(end - start for start, end in spans)
        if longest > config.SHOT_MAX_SECONDS + 0.5:
            warnings.append(
                f"beat {slot.beat_index} holds one shot for {longest:.1f}s - "
                f"step 4 sourced only {len(available)} usable asset(s) for it"
            )

        seen_here: dict[int, int] = {}
        for asset_index, (start, end) in zip(chosen, spans):
            repeat = seen_here.get(asset_index, 0)
            seen_here[asset_index] = repeat + 1
            plan.append({
                "beat_index": slot.beat_index,
                "asset_index": asset_index,
                "start": start,
                "end": end,
                "repeat": repeat,
            })

    # --- pass 2: put the cuts where they will be heard ---------------------
    # The beat grid claims the opening cuts first and the words take what is
    # left. The other order does not work: the word pass would find a nearer
    # pause for every cut the music had just placed, and the grid would end up
    # having done nothing at all.
    on_beat: list[int] = []
    if beats and config.SHOT_SNAP_TO_BEAT:
        on_beat = _snap_to_beats(plan, list(beats), words or [])
    snapped = 0
    if words and config.SHOT_SNAP_TO_WORDS:
        snapped = _snap_boundaries(plan, words, skip=on_beat)

    # --- pass 3: decide how each shot is played ----------------------------
    shots: list[Shot] = []
    for entry in plan:
        asset_index = entry["asset_index"]
        asset = assets[asset_index]
        duration = round(entry["end"] - entry["start"], 3)
        if duration <= 0:
            continue

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
                if entry["repeat"]:
                    in_point = round(max(0.0, min(
                        clip_length - duration,
                        in_point + entry["repeat"] * duration
                    )), 3)

        shots.append(Shot(
            index=len(shots),
            beat_index=entry["beat_index"],
            asset_index=asset_index,
            path=_asset_path(asset, asset_dir),
            kind=kind,
            start=entry["start"],
            duration=duration,
            source_start=in_point,
            looped=looped,
            reused=asset_index in used or entry["repeat"] > 0,
        ))
        used.add(asset_index)

    shots = _merge_identical_neighbours(shots)
    if shots:  # absorb the rounding drift rather than ending a frame short
        shots[-1].duration = round(total - shots[-1].start, 3)
    _plan_motion(shots, motion)

    if on_beat:
        warnings.append(
            f"{len(on_beat)} of the opening cuts were placed on the music's "
            f"beat grid (the first {config.SHOT_BEAT_WINDOW_SECONDS:g}s)"
        )
    if snapped:
        warnings.append(
            f"{snapped} of {len(plan) - 1} cuts were nudged onto a word "
            f"boundary (up to {config.SHOT_SNAP_MAX_SECONDS:g}s)"
        )

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


def _is_payoff_word(text: str) -> bool:
    """Is this the word the sentence was built to deliver?

    True for anything carrying a quantity - a digit, a number word, or a
    hyphenated compound of them ("ninety-five"). Everything else is a
    connective, and colouring connectives is how a highlight stops meaning
    anything. See _PAYOFF_NUMBER_WORDS for why digits alone are not enough.
    """
    clean = (text or "").strip().lower().strip(_PAYOFF_STRIP)
    if not clean:
        return False
    if _PAYOFF_DIGIT_RE.search(clean):
        return True
    parts = [part for part in clean.split("-") if part]
    return bool(parts) and all(part in _PAYOFF_NUMBER_WORDS for part in parts)


def _fit_pop_scale(text: str, font_file: Path | None, size: int,
                   wanted: int) -> int:
    """The largest pop `text` can take without reflowing onto another line.

    Scaling happens after libass has decided where the line breaks go, so a cue
    already filling the frame wraps for the length of the pop and snaps back.
    Measured on the salt reel at 112%: "50 QUADRILLION TONS" measures 872px of
    a 900px line at rest, goes to three rows at 1.30s and back to one at 1.40s,
    which reads as a rendering fault rather than as emphasis.

    Cues with room pop the full amount and the ones filling the frame pop less,
    or not at all. Capping is the only fix that costs nothing: fitting every cue
    to its popped size instead would shrink every caption in the reel by a
    tenth, permanently, for the sake of 90 milliseconds.
    """
    if wanted <= 100:
        return 100
    ink = 2 * config.CAPTION_OUTLINE_WIDTH + config.CAPTION_SHADOW_DEPTH
    usable = CAPTION_REFERENCE_WIDTH - 2 * config.CAPTION_MARGIN_H - ink
    width = _text_width(text, font_file, size)
    if width <= 0 or usable <= 0:
        return wanted
    return max(100, min(wanted, int(100 * usable / width)))


def _pop_tag(scale: int) -> str:
    """The transform that lands a cue oversized and settles it to 100%.

    `\\t(0,ms,...)` is timed from the start of the dialogue line it sits on, and
    every line here starts when its own segment does - so the pop rides the
    moment the text appears without anything having to know the cue's clock.

    Returns "" when there is nothing to draw, which is what keeps the ASS of a
    reel with CAPTION_POP off byte-for-byte what it always was.
    """
    if not config.CAPTION_POP or config.CAPTION_POP_MS <= 0:
        return ""
    if scale <= 100:
        return ""
    return (f"\\fscx{scale}\\fscy{scale}"
            f"\\t(0,{int(config.CAPTION_POP_MS)},\\fscx100\\fscy100)")


def _ass_override(size: int, *tags: str) -> str:
    """The single brace block a dialogue line opens with.

    One block rather than several: libass reads `{\\fs90}{\\fscx112}` and
    `{\\fs90\\fscx112}` the same way, but this file is read by people too and a
    line opening with three separate overrides is harder to check by eye.

    Empty when there is nothing to override - which is what keeps a reel with
    CAPTION_POP off and no oversized cue byte-for-byte what it always was.
    """
    parts = [f"\\fs{size}"] if size != config.CAPTION_FONT_SIZE else []
    parts += [tag for tag in tags if tag]
    return "{" + "".join(parts) + "}" if parts else ""


def _ass_escape(text: str) -> str:
    """Keep caption text from being read as ASS markup."""
    return (text.replace("\\", "").replace("{", "(").replace("}", ")")
                .replace("\n", " ").strip())


def _text_width(text: str, font_file: Path | None, size: int) -> float:
    """How wide `text` renders at `size`, in 1080x1920 caption pixels."""
    if font_file is not None:
        try:
            from PIL import ImageFont

            face = ImageFont.truetype(str(font_file), size)
            ascent, descent = face.getmetrics()
            # libass asks FreeType for a *real dimension* size: the number it is
            # given is ascender-to-descender, not the em square Pillow measures
            # against. Without this correction a heavy face like Arial Black
            # measures ~40% wider than it renders, and every cue would shrink
            # itself away from a frame edge it was never going to touch.
            return face.getlength(text) * size / max(1, ascent + descent)
        except Exception:
            pass
    return len(text) * size * 0.55   # no readable font file: assume a wide face


def _fit_font_size(text: str, font_file: Path | None, *, base: int,
                   ink: float, margin_h: int, lines: int = 1) -> int:
    """Shrink text that would otherwise run past the frame edge.

    A long single word - "bioluminescence" - cannot be wrapped, so it is scaled
    down instead. Measured with the real font when its file was found, and with
    a conservative width-per-character otherwise. `lines` is how many rows the
    text is allowed to wrap onto before it has to shrink.
    """
    # The outline grows the text on both sides and the shadow trails it on one,
    # so the glyphs get less than the bare margin-to-margin width. Leaving that
    # out lets text that measures as an exact fit wrap onto an extra line.
    usable = (CAPTION_REFERENCE_WIDTH - 2 * margin_h - ink) * max(1, lines)
    if usable <= 0 or not text:
        return base

    width = _text_width(text, font_file, base)
    if width <= usable:
        return base
    return max(24, int(base * usable / width))


def _cue_font_size(text: str, font_file: Path | None) -> int:
    """The size one caption cue is drawn at, shrunk if it would not fit."""
    return _fit_font_size(
        text, font_file,
        base=config.CAPTION_FONT_SIZE,
        ink=2 * config.CAPTION_OUTLINE_WIDTH + config.CAPTION_SHADOW_DEPTH,
        margin_h=config.CAPTION_MARGIN_H,
        lines=1,
    )


def _hook_card_font_size(text: str, font_file: Path | None) -> int:
    """The size the hook card is drawn at.

    Allowed two lines before it starts shrinking: the card is a headline of a
    handful of words, and wrapping one of them onto a second row is how a
    headline is normally set. Shrinking to keep it on one line would make the
    biggest text in the reel the smallest thing about it.
    """
    return _fit_font_size(
        text, font_file,
        base=config.HOOK_CARD_FONT_SIZE,
        ink=2 * config.HOOK_CARD_OUTLINE_WIDTH,
        margin_h=config.CAPTION_MARGIN_H,
        lines=2,
    )


def _hook_card_events(hook: str, seconds: float, font_file: Path | None,
                      fade_ms: int | None = None) -> tuple[list[str], list[str]]:
    """The style and the one dialogue line that draw the hook card.

    Returns (styles, events) - both empty when there is no card to draw, so the
    ASS of a reel without one is byte-for-byte what it always was.

    `fade_ms` overrides HOOK_CARD_FADE_MS. The cover passes 0: it is a single
    frame at t=0, which in the reel is the middle of the fade-in, and a cover
    with its headline at half opacity looks like a rendering fault.
    """
    text = _ass_escape(hook or "")
    if not text or seconds <= 0:
        return [], []
    if config.HOOK_CARD_UPPERCASE:
        text = text.upper()

    colour = _ass_colour(config.HOOK_CARD_COLOR)
    outline = _ass_colour(config.HOOK_CARD_OUTLINE_COLOR)
    size = _hook_card_font_size(text, font_file)

    # Alignment 8 is top-centre, and with it MarginV is measured down from the
    # top of the frame rather than up from the bottom - which is why the card
    # and the captions can share a file without ever meeting.
    styles = [
        f"Style: Hook,{{font}},{size},{colour},{colour},{outline},"
        f"&H80000000,0,0,0,0,100,100,0,0,1,"
        f"{config.HOOK_CARD_OUTLINE_WIDTH:g},0,8,"
        f"{config.CAPTION_MARGIN_H},{config.CAPTION_MARGIN_H},"
        f"{config.HOOK_CARD_MARGIN_V},1"
    ]

    fade = max(0, config.HOOK_CARD_FADE_MS if fade_ms is None else fade_ms)
    prefix = f"{{\\fad({fade},{fade})}}" if fade else ""
    events = [
        f"Dialogue: 0,{_ass_time(0.0)},{_ass_time(seconds)},Hook,,0,0,0,,"
        f"{prefix}{text}"
    ]
    return styles, events


def build_ass(cues: list[Cue], font_name: str, font_file: Path | None, *,
              hook: str = "", hook_seconds: float = 0.0,
              hook_fade_ms: int | None = None) -> str:
    """The subtitle file libass burns in.

    One dialogue line per cue in "plain" style; in "highlight" style one line
    per word, each showing the whole cue with the word being spoken tinted -
    which is what reads as the caption keeping up with the voice.

    `hook` adds the hook card over the opening: the same file, a second style,
    one more event. It rides along here rather than in the filter graph because
    libass is already being run over the picture - a drawtext pass for it would
    be a second text renderer with its own font handling and its own idea of
    where the margins are.
    """
    primary = _ass_colour(config.CAPTION_PRIMARY_COLOR)
    highlight = _ass_colour(config.CAPTION_HIGHLIGHT_COLOR)
    outline = _ass_colour(config.CAPTION_OUTLINE_COLOR)
    primary_inline = _ass_inline_colour(config.CAPTION_PRIMARY_COLOR)
    highlight_inline = _ass_inline_colour(config.CAPTION_HIGHLIGHT_COLOR)
    payoff_inline = _ass_inline_colour(config.CAPTION_PAYOFF_COLOR)

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
    ]

    card_styles, card_events = _hook_card_events(hook, hook_seconds, font_file,
                                                 hook_fade_ms)
    header += [style.format(font=font_name) for style in card_styles]
    header += [
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, "
        "Effect, Text",
    ]
    header += card_events

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
        payoff = [_is_payoff_word(word) for word in spoken]
        # Capped per cue: how much room a pop has is a property of this line's
        # text at this line's size, not of the reel.
        entry_pop = _pop_tag(_fit_pop_scale(joined, font_file, size,
                                            config.CAPTION_POP_SCALE))
        payoff_pop = _pop_tag(_fit_pop_scale(joined, font_file, size,
                                             config.CAPTION_PAYOFF_SCALE))

        if config.CAPTION_STYLE != "highlight" or len(spoken) == 1:
            if config.CAPTION_STYLE != "highlight":
                # "plain" means one colour for the whole cue, so the payoff
                # word gets the pop and not the tint - a second colour here
                # would make that setting a lie.
                body, pop = joined, entry_pop
            else:
                tint = payoff_inline if payoff[0] else highlight_inline
                body = f"{tint}{joined}{primary_inline}"
                pop = payoff_pop if payoff[0] else entry_pop
            lines.append(f"Dialogue: 0,{_ass_time(cue.start)},"
                         f"{_ass_time(cue.end)},Reel,,0,0,0,,"
                         f"{_ass_override(size, pop)}{body}")
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

        # A pop never interrupts a pop. Two accents inside one settle read as
        # a wobble rather than as two words being made, and "ninety-five
        # percent" - two payoff words in a row, often under 200ms apart - is
        # exactly the case that produces it. Same reasoning as
        # SFX_MIN_GAP_SECONDS: the effect is cheap and easy to overdo.
        settled = -1.0
        for position, _word in enumerate(spoken):
            tinted = " ".join(
                f"{payoff_inline if payoff[index] else highlight_inline}"
                f"{token}{primary_inline}"
                if index == position else token
                for index, token in enumerate(spoken)
            )
            # The cue pops as it lands, and a payoff word pops again - harder -
            # as it is spoken. Both can be true of the first word; the payoff is
            # the louder claim, so it takes the line.
            pop = (payoff_pop if payoff[position]
                   else entry_pop if position == 0 else "")
            if pop and points[position] < settled:
                pop = ""
            elif pop:
                settled = points[position] + config.CAPTION_POP_MS / 1000.0
            lines.append(
                f"Dialogue: 0,{_ass_time(points[position])},"
                f"{_ass_time(points[position + 1])},Reel,,0,0,0,,"
                f"{_ass_override(size, pop)}{tinted}"
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
        if shot.motion:
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


def _zoom_expression(shot: Shot, frames: int) -> str:
    """zoompan's `z`, moving from 1.0 to the shot's zoom or back the other way.

    Clamped at both ends rather than left to run on: `on` keeps counting past
    the frame the shot was planned for when a source hands over more frames
    than expected, and an unclamped ramp would carry on zooming into it.
    """
    zoom, travel = shot.motion_zoom, shot.motion_zoom - 1.0
    if shot.motion == "pull":
        return f"z='max({zoom:.4f}-{travel:.4f}*on/{frames},1.0000)'"
    return f"z='min(1+{travel:.4f}*on/{frames},{zoom:.4f})'"


def _shot_chain(shot: Shot, label_in: str, label_out: str) -> str:
    """The filter chain that turns one source into a frame-shaped segment."""
    width, height, fps = config.VIDEO_WIDTH, config.VIDEO_HEIGHT, config.VIDEO_FPS

    if shot.motion:
        frames = max(1, int(round(shot.duration * fps)))
        # Oversample first: zoompan quantises its zoom to whole input pixels, so
        # a larger source makes the move smooth instead of steppy.
        big_w, big_h = int(width * MOTION_OVERSAMPLE) // 2 * 2, \
            int(height * MOTION_OVERSAMPLE) // 2 * 2
        steps = [
            f"scale={big_w}:{big_h}:force_original_aspect_ratio=increase"
            f":flags=lanczos",
            f"crop={big_w}:{big_h}",
        ]
        if shot.kind == "image":
            # One input frame, and zoompan makes the whole shot out of it.
            hold = f"d={frames}"
        else:
            # A clip arrives at whatever rate it was shot at, and zoompan with
            # d=1 emits one output frame per *input* frame. Settle the rate
            # before it starts counting, or a 24fps source hands back 24 frames
            # labelled as 30 and the shot plays a quarter too fast. Measured
            # against the static path on 24fps and 30fps sources, this ordering
            # returns the identical frame count; the other one does not.
            steps.append(f"fps={fps}")
            hold = "d=1"
        steps.append(f"zoompan={_zoom_expression(shot, frames)}"
                     f":{hold}:s={width}x{height}:fps={fps}"
                     f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'")
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


def _music_chains(audio_index: int, music_index: int,
                  narration_chain: list[str], total: float,
                  out_label: str = "aout") -> list[str]:
    """The bed, and how it goes under the voice.

    Ducking needs the narration twice over - once as the thing you actually
    hear, once as the key telling the compressor when to pull the music down -
    so it is split rather than decoded twice.
    """
    ducking = config.MUSIC_DUCKING and config.MUSIC_DUCK_RATIO > 1
    voice = ",".join(narration_chain)
    chains = [f"[{audio_index}:a]{voice}"
              + ("," + "asplit=2[narr][duckkey]" if ducking else "[narr]")]

    # The input is fed in with -stream_loop, so a track shorter than the reel
    # repeats; atrim is what makes that infinite stream finite again.
    bed = [
        f"atrim=end={total:.3f}",
        f"aresample={config.AUDIO_SAMPLE_RATE}",
        f"loudnorm=I={config.MUSIC_LOUDNESS_TARGET:g}"
        f":TP={config.AUDIO_LOUDNESS_PEAK:g}"
        f":LRA={config.AUDIO_LOUDNESS_RANGE:g}",
    ]
    if config.MUSIC_FADE_IN_SECONDS > 0:
        bed.append(f"afade=t=in:st=0:d={config.MUSIC_FADE_IN_SECONDS:g}")
    if config.MUSIC_FADE_OUT_SECONDS > 0:
        start = max(0.0, total - config.MUSIC_FADE_OUT_SECONDS)
        bed.append(f"afade=t=out:st={start:.3f}"
                   f":d={config.MUSIC_FADE_OUT_SECONDS:g}")
    chains.append(f"[{music_index}:a]" + ",".join(bed) + "[bed]")

    mixed = "bed"
    if ducking:
        chains.append(
            f"[bed][duckkey]sidechaincompress="
            f"threshold={config.MUSIC_DUCK_THRESHOLD:g}"
            f":ratio={config.MUSIC_DUCK_RATIO:g}"
            f":attack={config.MUSIC_DUCK_ATTACK_MS:g}"
            f":release={config.MUSIC_DUCK_RELEASE_MS:g}"
            f":makeup=1[ducked]"
        )
        mixed = "ducked"

    # normalize=0 is not optional: amix divides by the input count by default,
    # so leaving it out would halve the narration the moment music was added.
    chains.append(f"[narr][{mixed}]amix=inputs=2:normalize=0:duration=first"
                  f"[{out_label}]")
    return chains


def build_filter_graph(shots: list[Shot], audio_index: int, total: float,
                       narration: float, ass_path: Path | None,
                       fonts_dir: Path | None,
                       music_index: int | None = None,
                       sfx_cues: list[SfxCue] | None = None,
                       sfx_index: int = 0) -> tuple[str, str]:
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

    effect_chains, effect_labels = _sfx_chains(sfx_cues or [], sfx_index)
    # With effects to add, the voice-and-music mix is no longer the last word.
    bed = "abed" if effect_labels else "aout"

    if music_index is None:
        chains.append(f"[{audio_index}:a]" + ",".join(audio) + f"[{bed}]")
    else:
        chains += _music_chains(audio_index, music_index, audio, total,
                                out_label=bed)

    if effect_labels:
        chains += effect_chains
        joined = "".join(f"[{label}]" for label in effect_labels)
        # The effects go on top of the finished mix rather than through the
        # ducking with it. They are transients whose whole job is to punch, and
        # a compressor keyed off the narration would pull each one down at
        # exactly the moment it fired.
        chains.append(
            f"[{bed}]{joined}amix=inputs={1 + len(effect_labels)}"
            f":normalize=0:duration=first[aout]"
        )

    return ";".join(chains), stage


def render(shots: list[Shot], audio_path: Path, output_path: Path, *,
           total: float, narration: float, ass_path: Path | None,
           fonts_dir: Path | None, work_dir: Path, music: Track | None = None,
           sfx: list[SfxCue] | None = None,
           overwrite: bool = True) -> list[str]:
    """Encode the reel. One ffmpeg pass: no intermediates, one generation loss."""
    if not shots:
        raise NoFootageError("there are no shots to render")

    inputs: list[str] = []
    for shot in shots:
        inputs += _shot_inputs(shot)
    audio_index = len(shots)
    inputs += ["-i", str(audio_path)]

    music_index = None
    if music is not None:
        music_index = audio_index + 1
        # -stream_loop rather than the aloop filter: a track shorter than the
        # reel repeats by re-reading the file, instead of holding every sample
        # of it in memory to loop from.
        inputs += ["-stream_loop", "-1", "-i", str(music.path)]

    # One input per placement, not per file: the same whoosh landing on four
    # cuts is four inputs, because each one needs its own delay. They are a few
    # hundred milliseconds each, so the decode costs nothing worth avoiding.
    sfx_cues = list(sfx or [])
    sfx_index = (music_index if music_index is not None else audio_index) + 1
    for cue in sfx_cues:
        inputs += ["-i", str(cue.sfx.path)]

    graph, stage = build_filter_graph(
        shots, audio_index, total, narration, ass_path, fonts_dir, music_index,
        sfx_cues=sfx_cues, sfx_index=sfx_index,
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
# The cover frame
#
# The still Instagram shows in the grid, on the profile and in search. The reel
# earns the watch; the cover earns the tap that gets it played at all.
#
# It is deliberately NOT a frame lifted out of reel.mp4. The running captions
# are burned into every frame of that file - measured on a real run, the first
# cue is up at 0.079s and there is one on screen essentially continuously - so
# a grab from the finished video puts three words out of the middle of a
# sentence across the bottom of the tile. The cover is rebuilt from the source
# asset instead, with the edit's own scale and crop, and the hook card redrawn
# over it through the same libass path the video uses. That way the card on the
# cover cannot drift from the card in the reel: it is built by the same code.
# ---------------------------------------------------------------------------

#: Line height as a multiple of font size, for working out how far down the
#: frame the hook card reaches. Approximate on purpose - it is used for a
#: clearance check with a lot of room in it, not for layout.
_CARD_LINE_HEIGHT = 1.2

#: Rows to assume the card wraps onto for that check. `_hook_card_font_size`
#: fits to two before it starts shrinking, but that governs the shrink, not
#: libass's wrapping - a real card measured here came out on three. The check
#: is a clearance, so it takes the taller number.
_CARD_ASSUMED_LINES = 3

#: Trailing "-12345678" on a stock provider's URL slug: the asset id, not words.
_SLUG_ID_RE = re.compile(r"[-_]\d+$")

#: Words a stock library puts in its own titles that describe the file rather
#: than the picture. "vertical video of a jellyfish" is alt text about a video.
_SLUG_NOISE = ("free stock video", "stock footage", "stock video",
               "vertical video of", "vertical video", "video of", "footage of",
               "free video", "video", "footage", "clip", "vertical")

#: Words that own a bare "t" once the slug has dropped the apostrophe. A
#: bare "t" is not treated like a bare "s", because the two orphans are not
#: the same problem. "s" is left over from a possessive and a stock library
#: has almost no other use for it; "t" heads two of the commonest words in
#: one - "t-shirt" and "t-cell" arrive here as "t shirt" and "t cell" - so
#: attaching every orphan "t" backwards turns "a t shirt" into "a't shirt"
#: and "vitamin c and t cells" into "and't cells". Every English "t"
#: contraction is "n't", which makes the set of stems closed and short
#: enough to name, so the rule can be about contractions rather than about
#: the letter.
_CONTRACTION_STEMS = frozenset((
    "ain", "aren", "can", "couldn", "didn", "doesn", "don", "hadn", "hasn",
    "haven", "isn", "mightn", "mustn", "needn", "shan", "shouldn", "wasn",
    "weren", "won", "wouldn",
))


def parse_aspect(text: str) -> tuple[int, int]:
    """"4:5" -> (4, 5). Anything unreadable falls back to the frame's own."""
    match = re.fullmatch(r"\s*(\d+)\s*[:xX/]\s*(\d+)\s*", str(text or ""))
    if match and int(match[1]) > 0 and int(match[2]) > 0:
        return int(match[1]), int(match[2])
    return config.VIDEO_WIDTH, config.VIDEO_HEIGHT


def safe_band(aspect: str) -> tuple[float, float]:
    """Top and bottom of what survives a centre crop of the frame to `aspect`.

    In CAPTION_REFERENCE_* pixels, which is what the hook card's margins are
    expressed in - so the two can be compared without knowing the output size.
    """
    across, down = parse_aspect(aspect)
    height = min(CAPTION_REFERENCE_HEIGHT,
                 CAPTION_REFERENCE_WIDTH * down / max(1, across))
    margin = (CAPTION_REFERENCE_HEIGHT - height) / 2
    return margin, margin + height


def card_clears_crop(card: str, font_file: Path | None,
                     aspect: str) -> tuple[bool, str]:
    """Does the hook card survive being centre-cropped to `aspect`?

    The card is top-anchored at HOOK_CARD_MARGIN_V, so what a centre crop takes
    is the top of it - the first line of the headline, which is the half that
    matters. The bottom is estimated from _CARD_ASSUMED_LINES rather than
    measured, because this is a clearance check and the wrong way to be wrong
    is optimistic.
    """
    top, bottom = safe_band(aspect)
    size = _hook_card_font_size(card, font_file)
    card_top = float(config.HOOK_CARD_MARGIN_V)
    card_bottom = card_top + _CARD_ASSUMED_LINES * size * _CARD_LINE_HEIGHT
    if card_top < top:
        return False, (f"the hook card sits {card_top:.0f}px down and a centre "
                       f"crop to {aspect} starts at {top:.0f}px, so its first "
                       f"line would be cut off")
    if card_bottom > bottom:
        return False, (f"the hook card reaches {card_bottom:.0f}px down and a "
                       f"centre crop to {aspect} ends at {bottom:.0f}px")
    return True, ""


def _describe(text: str) -> str:
    """Tidy a provider's own words for a clip into something readable.

    Handles a URL slug ("medusa-blue-ocean-13693981"), Pexels' alt text and
    Pixabay's comma-separated tags with the same treatment, because all three
    arrive as the same kind of thing: words about the picture, loosely punctuated.
    """
    words = _SLUG_ID_RE.sub("", str(text or "").strip().strip("/"))
    words = words.replace("-", " ").replace("_", " ")
    words = re.sub(r"\s*,\s*", ", ", words)
    words = " ".join(words.split())

    # A slug drops the apostrophe before it is ever split, so Pexels'
    # "...looking-the-patient-s-results..." arrives here as "the patient s
    # results" and ships in alt text, which is the one line on a post written
    # to be read aloud. An orphan "s" belongs to the word in front of it; an
    # orphan "t" only does when that word is a contraction stem, for the
    # reason _CONTRACTION_STEMS gives.
    words = re.sub(r"(?<=[^\W\d_])\s+s\b", "'s", words)
    words = re.sub(
        r"\b(\w+)\s+t\b",
        lambda m: (f"{m[1]}'t" if m[1].lower() in _CONTRACTION_STEMS
                   else m[0]),
        words,
    )

    lowered = words.lower()
    for noise in _SLUG_NOISE:
        if lowered.startswith(noise + " "):
            words = words[len(noise) + 1:]
        elif lowered.endswith(" " + noise):
            words = words[: -len(noise) - 1]
        lowered = words.lower()
    return " ".join(words.split()).strip(" ,.")


def _cover_depiction(asset: dict) -> tuple[str, str]:
    """What the cover actually shows, and where that description came from.

    Ordered by how close each source sits to the picture on screen. Step 4's
    `image_prompt` is deliberately NOT in this list, and it is the one that
    looks like it should be: it describes what the beat ASKED for, and stock
    search does not always answer. On a real run "a snailfish swims in the deep
    ocean" was filled by a clip of a jellyfish, and "a dark, high-pressure
    environment" by a diver in bright, clear water. Alt text is read by people
    who cannot check it against the image, so it is built from the words the
    provider put on the file that was actually used.
    """
    for field, name in (("title", "provider_title"),
                        ("source_url", "source_url"),
                        ("query", "query")):
        raw = asset.get(field) or ""
        if name == "source_url":
            raw = str(raw).rstrip("/").rsplit("/", 1)[-1]
        described = _describe(raw)
        if described:
            return described, name
    return "", "none"


def cover_score(shot: Shot, probe: dict, verdict: dict | None = None) -> float:
    """Rank a shot as a candidate for the cover. Higher is better.

    A different question from `score_candidate` in step 4, which asks whether a
    clip can carry a beat. This asks whether one frame of it, with no motion and
    no sound, is worth a stranger's tap.

    `verdict` is step 5b's answer to exactly that question, from having been
    shown the frame - see `look.judge_covers`. Everything below it is a proxy
    standing in for a look that until now nobody took: a long slot means the
    beat had footage worth holding on, native vertical means no crop, more
    pixels means fewer artefacts. All three are true and none of them is about
    what the frame shows, which is why the look outweighs any one of them and
    a veto outweighs all of them together. It stays a term rather than a gate
    because the model sees one frame and the proxies know things about the clip
    that one frame does not: with no verdict, the score is what it always was.
    """
    score = 0.0

    if verdict:
        if verdict.get("ok", True):
            score += config.COVER_LOOK_WEIGHT * (
                _clamped_grade(verdict) / look.GRADE_MAX
            )
        else:
            # Not disqualifying. A run where the model dislikes every candidate
            # must still ship a cover, and Instagram takes the first frame of
            # the video when it is given none - which is a caption mid-sentence
            # over whatever was on screen at 0.0s, and worse than all of them.
            score -= config.COVER_LOOK_VETO

    # The assembler gave this shot a long slot, which means the beat it belongs
    # to had footage worth holding on.
    score += 3.0 * min(shot.duration / max(0.1, config.SHOT_TARGET_SECONDS), 1.0)

    width = int(probe.get("width") or 0)
    height = int(probe.get("height") or 0)
    if width and height:
        # Native vertical needs no crop, so the cover is the whole picture the
        # way it was shot rather than a slice out of the middle of it.
        wanted = config.VIDEO_WIDTH / config.VIDEO_HEIGHT
        if width / height <= wanted * 1.02:
            score += 2.0
        # Resolution headroom, capped the way step 4 caps it.
        score += min(height / max(1, config.VISUAL_MIN_HEIGHT), 2.0)

    if shot.looped:
        score -= 2.5      # too short for its slot: a scrap, not a hero shot
    if shot.reused:
        score -= 2.0      # already on screen elsewhere; nothing distinctive

    # Mild preference for the top of the reel, where the promise is.
    score -= shot.start / max(1.0, config.COVER_SEARCH_SECONDS)
    return round(score, 4)


def _clamped_grade(verdict: dict) -> float:
    """A verdict's grade, trusted no further than the scale it is on."""
    try:
        return max(0.0, min(float(look.GRADE_MAX), float(verdict.get("grade") or 0)))
    except (TypeError, ValueError):
        return 0.0


def cover_candidates(shots: list[Shot], *, card: str, card_seconds: float,
                     font_file: Path | None, source: str | None = None,
                     warnings: list[str] | None = None) -> dict:
    """Which shots the cover could be taken from, and under what. Pure.

    Split out of `plan_cover` so the shortlist can be rendered and looked at
    before it is ranked. The two callers have to agree exactly: a preview of a
    shot that was never in the running spends a model call on nothing, and one
    rendered without the headline card the reel puts over it would be judged on
    a frame the reel does not contain.

    Returns the shortlist and everything decided on the way to it - which mode
    was honoured, whether there is a card and whether it survives the crop.
    """
    notes = warnings if warnings is not None else []

    mode = str(source or config.COVER_SOURCE).strip().lower()
    if mode not in ("auto", "hook_card", "footage"):
        notes.append(f"unknown COVER_SOURCE {mode!r}; used 'auto'")
        mode = "auto"

    aspect = config.COVER_SAFE_ASPECT
    has_card = bool(card) and card_seconds > 0
    clears, clearance = (card_clears_crop(card, font_file, aspect)
                         if has_card else (False, ""))

    window = max(0.0, config.COVER_SEARCH_SECONDS)
    early = [shot for shot in shots if shot.start < window] or [shots[0]]

    if mode == "hook_card" and not has_card:
        notes.append("COVER_SOURCE=hook_card but this reel has no hook card; "
                     "took the cover from the footage instead")
        mode = "auto"

    want_card = has_card and (
        mode == "hook_card" or (mode == "auto" and clears)
    )
    if mode == "auto" and has_card and not clears:
        notes.append(f"cover taken from the footage rather than the hook card: "
                     f"{clearance}")

    if want_card:
        # The card usually spans more than one shot. Take the frame from the
        # best of the ones underneath it, so the picture behind the headline is
        # chosen rather than whatever happened to be running at that instant.
        under = [shot for shot in shots if shot.start < card_seconds]
        # A shot that only catches the last fraction of the card is a bad place
        # to take the frame from: there is barely any of it inside the window,
        # and what there is sits under a card that is on its way out.
        solid = [shot for shot in under
                 if min(shot.end, card_seconds) - shot.start >= 2 * _COVER_EDGE_SECONDS]
        pool = solid or under or [shots[0]]
        reason = "the reel opens on a hook card"
        if mode == "hook_card" and not clears:
            reason += f", and {clearance} (COVER_SOURCE=hook_card kept it anyway)"
        else:
            reason += f", and it clears a centre crop to {aspect}"
    else:
        pool = [shot for shot in early if shot.start >= card_seconds] or early
        reason = "the strongest footage shot near the top of the reel"

    return {"pool": pool, "want_card": want_card, "reason": reason,
            "mode": mode, "has_card": has_card, "clears": clears,
            "aspect": aspect}


def cover_frame_time(shot: Shot, *, card_seconds: float,
                     want_card: bool) -> tuple[float, float]:
    """Which frame of `shot` the cover is: (time in the reel, time in the file).

    Stills have one frame; clips are seeked inside the same in-point the edit
    uses, so the cover is a frame the reel genuinely contains.
    """
    edge = min(_COVER_EDGE_SECONDS, shot.duration / 3)
    offset = min(max(shot.duration * _COVER_SHOT_POSITION, edge),
                 max(edge, shot.duration - edge))
    if want_card:
        # Has to land while the card is actually up, or the cover is a frame of
        # footage with a headline pasted on that the viewer never sees.
        offset = min(offset, max(edge, card_seconds - shot.start - edge))
        offset = max(0.0, offset)
    source_time = 0.0 if shot.kind == "image" else round(
        shot.source_start + offset, 3
    )
    return round(shot.start + offset, 3), source_time


def cover_label(shot: Shot) -> str:
    """What a candidate frame is called when it is put in front of the model.

    The shot index rather than a letter, so a verdict read back into
    assembly.json lines up against the timeline beside it without counting.
    """
    return f"shot {shot.index}"


def plan_cover(shots: list[Shot], manifest: dict, probes: dict[int, dict], *,
               card: str, card_seconds: float, font_file: Path | None,
               source: str | None = None,
               looks: dict[str, dict] | None = None,
               warnings: list[str] | None = None) -> dict | None:
    """Choose the cover frame. Decides everything; writes nothing.

    Split from `write_cover` the way `plan_timeline` is split from `render`, so
    the choice can be tested without an encoder and honoured by --dry-run.

    `looks` is step 5b's verdict per candidate, keyed by `cover_label`, from
    having been shown the rendered frames. It is passed in rather than fetched
    here for the same reason `probes` is: this function stays pure, and a dry
    run gets the whole decision without a model call or an encoder.

    Returns the cover record for assembly.json, or None when covers are off or
    there is nothing to take one from.
    """
    notes = warnings if warnings is not None else []
    if not config.COVER_ENABLED or not shots:
        return None

    verdicts = looks or {}
    choice = cover_candidates(shots, card=card, card_seconds=card_seconds,
                              font_file=font_file, source=source,
                              warnings=notes)
    pool, want_card = choice["pool"], choice["want_card"]
    aspect, reason = choice["aspect"], choice["reason"]

    chosen = max(pool, key=lambda shot: (
        cover_score(shot, probes.get(shot.asset_index, {}),
                    verdicts.get(cover_label(shot))),
        -shot.index,
    ))
    probe = probes.get(chosen.asset_index, {})
    verdict = verdicts.get(cover_label(chosen))

    if verdict and not verdict.get("ok", True):
        # Either every candidate was vetoed, or the vetoed one still outscored
        # the rest. Both mean the cover that ships is one the model said not to
        # ship, which belongs in the warnings and not only in the JSON.
        why = str(verdict.get("why") or "").strip()
        notes.append(
            f"the cover look would not ship any candidate; kept "
            f"{cover_label(chosen)}" + (f" ({why})" if why else "")
        )

    time, source_time = cover_frame_time(chosen, card_seconds=card_seconds,
                                         want_card=want_card)

    assets = manifest.get("assets") or []
    asset = (assets[chosen.asset_index]
             if 0 <= chosen.asset_index < len(assets) else {})
    depicts, depicts_source = _cover_depiction(asset)
    if not depicts:
        depicts = _describe(manifest.get("topic") or "")
        depicts_source = "topic" if depicts else "none"
        notes.append("nothing describes the cover's footage - step 6 has only "
                     "the topic to write alt text from")

    return {
        "path": None,
        "filename": config.COVER_FILENAME,
        "written": False,
        "source": "hook_card" if want_card else "footage",
        "mode": choice["mode"],
        "reason": reason,
        "time": time,
        "source_time": source_time,
        "shot_index": chosen.index,
        "beat_index": chosen.beat_index,
        "asset_index": chosen.asset_index,
        "file": chosen.path.name,
        "kind": chosen.kind,
        "hook_card_text": card if want_card else "",
        "depicts": depicts,
        "depicts_source": depicts_source,
        "safe_aspect": aspect,
        "card_in_safe_band": choice["clears"] if choice["has_card"] else None,
        "score": cover_score(chosen, probe, verdict),
        # What the model said about the frame that shipped, and about the ones
        # it beat. Both, because "this cover graded 3" is worth much less than
        # "this cover graded 3 and the two it beat graded 1" - the second says
        # whether the look changed the choice, which is the only way to find
        # out later whether it was worth the call it costs.
        "look": verdict,
        "look_candidates": [
            {"label": cover_label(shot), "shot_index": shot.index,
             **(verdicts.get(cover_label(shot)) or {})}
            for shot in pool
        ] if verdicts else [],
        "width": config.VIDEO_WIDTH,
        "height": config.VIDEO_HEIGHT,
        "bytes": 0,
    }


def _cover_frame_filters(width: int, height: int) -> list[str]:
    """The scale-and-crop the cover is built with. One definition, two callers.

    `write_cover` and `preview_cover_frames` have to agree or the look is
    taken on a picture the cover is not: the same principle the hook card
    already follows by being drawn through the same `build_ass` path as the
    video's.
    """
    return [
        f"scale={width}:{height}:force_original_aspect_ratio=increase"
        f":flags=lanczos",
        f"crop={width}:{height}",
        "setsar=1",
    ]


def _render_cover_frame(shot: Shot, source_time: float, card_text: str,
                        output: Path, *, font_name: str,
                        font_file: Path | None, fonts_dir: Path | None,
                        width: int, height: int, quality: int,
                        ass_path: Path) -> list[str]:
    """One frame of one shot, cropped and captioned the way the cover is.

    Extracted from `write_cover` so candidate frames can be rendered before the
    cover is chosen, through this exact code rather than something that looks
    like it.
    """
    args = [ffmpeg_binary(), "-hide_banner", "-nostdin", "-y"]
    if shot.kind != "image" and source_time > 0:
        args += ["-ss", f"{source_time:.3f}"]
    args += ["-i", str(shot.path)]

    steps = _cover_frame_filters(width, height)

    if card_text:
        # A card-only subtitle file, built by the same function that builds the
        # reel's - so the cover's headline wraps, shrinks and sits exactly where
        # the video's does. No fade: this is one frame at t=0, which in the reel
        # is the middle of the fade-in. libass carries the reference resolution
        # as PlayRes and scales to whatever it is rendered at, so a preview at a
        # smaller size gets the card at the same proportion of the frame.
        ass_path.write_text(
            build_ass([], font_name, font_file,
                      hook=card_text, hook_seconds=1.0, hook_fade_ms=0),
            encoding="utf-8",
        )
        burn = f"subtitles=filename={_filter_path(ass_path)}"
        if fonts_dir is not None:
            burn += f":fontsdir={_filter_path(fonts_dir)}"
        steps.append(burn)

    args += ["-vf", ",".join(steps), "-frames:v", "1", "-update", "1",
             "-q:v", str(quality), str(output)]

    try:
        done = subprocess.run(
            args, capture_output=True, text=True, errors="replace",
            timeout=FFMPEG_TIMEOUT, cwd=str(config.BASE_DIR),
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return [f"could not write {output.name}: {exc}"]

    if done.returncode != 0 or not output.is_file():
        return [f"ffmpeg could not write {output.name}: "
                + (done.stderr or "")[-400:].strip()]
    return []


def write_cover(cover: dict, shots: list[Shot], destination: Path, *,
                font_name: str, font_file: Path | None,
                fonts_dir: Path | None) -> list[str]:
    """Extract the planned cover frame. One ffmpeg call, one frame out.

    Mutates `cover` with where it landed and how big it is. Never fatal: a reel
    with no cover is still a reel, and the .mp4 is already on disk by here.
    """
    shot = next((s for s in shots if s.index == cover["shot_index"]), None)
    if shot is None:
        return ["the planned cover shot is not on the timeline"]

    output = destination / config.COVER_FILENAME
    problems = _render_cover_frame(
        shot, cover["source_time"], cover["hook_card_text"], output,
        font_name=font_name, font_file=font_file, fonts_dir=fonts_dir,
        width=config.VIDEO_WIDTH, height=config.VIDEO_HEIGHT,
        quality=config.COVER_JPEG_QUALITY,
        ass_path=destination / COVER_ASS_NAME,
    )
    if problems:
        return problems

    cover["path"] = str(output.resolve())
    cover["written"] = True
    cover["bytes"] = output.stat().st_size
    return []


# ---------------------------------------------------------------------------
# Frames for step 5b
#
# The only two places this project turns a video back into pictures. Both write
# into a temporary directory and hand bytes to `look.py`, which is what keeps
# the run directory free of a dozen JPEGs nobody will open: what is worth
# keeping is the verdict, and that goes into assembly.json.
# ---------------------------------------------------------------------------

def _look_size() -> tuple[int, int]:
    """The size a frame is sent at, keeping the output's own aspect."""
    width = max(64, int(config.LOOK_FRAME_WIDTH))
    height = max(2, round(width * config.VIDEO_HEIGHT / config.VIDEO_WIDTH))
    return width, height - (height % 2)


def preview_cover_frames(pool: list[Shot], *, card: str, card_seconds: float,
                         want_card: bool, font_name: str,
                         font_file: Path | None, fonts_dir: Path | None,
                         ) -> tuple[list[tuple[str, bytes]], list[str]]:
    """Render the candidate covers so they can be looked at before choosing.

    Returns (labelled JPEG bytes, warnings). Candidates past
    COVER_LOOK_MAX_CANDIDATES are dropped from the end of `pool`, which arrives
    in timeline order - so what is dropped is the latest in the reel, and the
    top of the reel is where `cover_score` already prefers to look.

    A shot that will not render is skipped rather than fatal. It keeps its
    blind score and competes ungraded, and `look.judge_covers` says so.
    """
    if not pool:
        return [], []

    width, height = _look_size()
    cap = max(1, int(config.COVER_LOOK_MAX_CANDIDATES))
    shortlist = pool[:cap]
    notes: list[str] = []
    if len(pool) > cap:
        notes.append(f"the cover look saw {cap} of {len(pool)} candidates "
                     f"(COVER_LOOK_MAX_CANDIDATES)")

    frames: list[tuple[str, bytes]] = []
    with tempfile.TemporaryDirectory(prefix="cover-look-") as work:
        work_dir = Path(work)
        for shot in shortlist:
            _, source_time = cover_frame_time(
                shot, card_seconds=card_seconds, want_card=want_card
            )
            output = work_dir / f"candidate-{shot.index}.jpg"
            problems = _render_cover_frame(
                shot, source_time, card if want_card else "", output,
                font_name=font_name, font_file=font_file, fonts_dir=fonts_dir,
                width=width, height=height,
                quality=config.LOOK_JPEG_QUALITY,
                ass_path=work_dir / f"candidate-{shot.index}.ass",
            )
            if problems:
                notes += problems
                continue
            frames.append((cover_label(shot), output.read_bytes()))
    return frames, notes


def narration_lines(manifest: dict) -> list[tuple[float, float, str]]:
    """The beats as (start, end, text), for step 5b to align frames against.

    The beats are the right grain for this and the alignment is not: a caption
    cue is three words long, so a frame would land against a fragment with no
    claim in it, while a beat is the sentence whose claim the footage was
    chosen to carry. Step 4 already searched per beat, so a beat is also the
    unit whose footage can be wrong.

    Beats missing their timings are skipped rather than guessed at - a wrong
    time is worse here than a missing one, because it hands the grader a line
    that was not playing and invites it to reject footage that matched.
    """
    rows: list[tuple[float, float, str]] = []
    for beat in manifest.get("beats") or []:
        if not isinstance(beat, dict):
            continue
        try:
            start = float(beat["start"])
            end = float(beat["end"])
        except (KeyError, TypeError, ValueError):
            continue
        text = " ".join(str(beat.get("text") or "").split())
        if text:
            rows.append((start, end, text))
    return rows


#: How far a sampled frame is kept from either end of the shot it belongs to.
#: The timeline in assembly.json is the plan, and the render drifts from it:
#: every shot is encoded as a whole number of frames, the rounding is per shot
#: and it accumulates. On the giant squid reel of 2026-09-19 the plan put the
#: cut at 20.61s and the render made it at about 20.50s - three frames early,
#: nine shots in.
#:
#: That is enough to make a verdict name the wrong clip. The frame sampled at
#: 20.53s came back "a small shark swimming over a reef", graded 0 of 3, and
#: the timeline says 18.90-20.61s is "serene ocean waves in sunlight". Anything
#: reading the grade back - a person, or one day a feedback edge - would have
#: blamed the ocean and left the shark alone. Sampling inside this margin
#: instead makes the label and the timeline agree.
_LOOK_SHOT_EDGE_SECONDS = 0.25


def _shot_at(shots: list[Shot] | None, at: float) -> Shot | None:
    """The shot covering `at`, or None when there is no shot list to ask."""
    for shot in shots or []:
        if shot.start <= at < shot.start + shot.duration:
            return shot
    # Past the last shot is the tail, which is nobody's footage.
    return None


def _inside_shot(shot: Shot, at: float) -> float:
    """Pull `at` clear of both ends of its shot, if the shot is long enough.

    A shot shorter than twice the margin has no safe interior, so it is
    sampled at its midpoint: the best available answer, and still the furthest
    point from either cut.
    """
    end = shot.start + shot.duration
    if shot.duration <= _LOOK_SHOT_EDGE_SECONDS * 2:
        return round(shot.start + shot.duration / 2, 3)
    return round(min(max(at, shot.start + _LOOK_SHOT_EDGE_SECONDS),
                     end - _LOOK_SHOT_EDGE_SECONDS), 3)


def _place_verdicts(record: dict, placement: dict[str, dict]) -> None:
    """Name the shot each of step 5b's frame verdicts is about, in place.

    A timestamp only names a shot if you go back to `timeline` and `timeline`
    agrees with the render, and near a cut it does not - see
    `_LOOK_SHOT_EDGE_SECONDS`. Writing the answer down where the verdict is
    saves every later reader from re-deriving it, and from re-deriving it
    wrongly in exactly the cases that matter, which are the ones where a shot
    changed.

    A verdict with no placement is left alone rather than given a null shot:
    the two are different, and only one of them is "this frame was not in any
    shot". Labels the model invents are already dropped by `look.grade_reel`,
    so what reaches here and misses is a frame taken with no shot list.
    """
    for entry in record.get("frames") or []:
        found = placement.get(entry.get("frame"))
        if found:
            entry.update(found)


def sample_reel_frames(video_path: Path, duration: float, *,
                       count: int | None = None,
                       shots: list[Shot] | None = None,
                       placement: dict[str, dict] | None = None,
                       ) -> tuple[list[tuple[str, bytes]], list[str]]:
    """Take evenly spaced frames out of the finished reel.

    Sampled at the midpoint of each equal slice rather than at the boundaries,
    so neither the first nor the last frame is taken: the reel opens on a
    fade-in and ends on a tail, and both would be graded as footage when they
    are transitions.

    Labels are timestamps, because that is what makes a verdict actionable -
    "dark at 22.4s" names a shot on the timeline in assembly.json, where
    "frame 4" names nothing. `shots` is what makes that true rather than
    nearly true: a sample landing within `_LOOK_SHOT_EDGE_SECONDS` of a cut is
    moved to the nearest point inside its own shot, because the cut in the
    rendered file is not exactly where the plan says it is.

    `placement` is an out-parameter, filled with `{label: {"shot", "beat"}}` -
    which shot each frame actually shows, decided here rather than by anyone
    reading timestamps back out of the manifest afterwards. It leaves this way
    for the reason the screener's `subject_keys` does: the callers that want
    the frames mostly do not want this, and a third element in the return
    would make every one of them carry it.
    """
    wanted = int(config.REEL_LOOK_FRAMES if count is None else count)
    if wanted < 1 or duration <= 0 or not Path(video_path).is_file():
        return [], []

    width, height = _look_size()
    frames: list[tuple[str, bytes]] = []
    notes: list[str] = []

    with tempfile.TemporaryDirectory(prefix="reel-look-") as work:
        work_dir = Path(work)
        for position in range(wanted):
            at = duration * (position + 0.5) / wanted
            shot = _shot_at(shots, at)
            if shot is not None:
                at = _inside_shot(shot, at)
            output = work_dir / f"frame-{position}.jpg"
            args = [
                ffmpeg_binary(), "-hide_banner", "-nostdin", "-y",
                "-ss", f"{at:.3f}", "-i", str(video_path),
                "-vf", f"scale={width}:{height}:flags=lanczos",
                "-frames:v", "1", "-update", "1",
                "-q:v", str(config.LOOK_JPEG_QUALITY), str(output),
            ]
            try:
                done = subprocess.run(
                    args, capture_output=True, text=True, errors="replace",
                    timeout=FFMPEG_TIMEOUT, cwd=str(config.BASE_DIR),
                )
            except (OSError, subprocess.SubprocessError) as exc:
                notes.append(f"could not sample the reel at {at:.1f}s: {exc}")
                continue
            if done.returncode != 0 or not output.is_file():
                notes.append(f"ffmpeg could not sample the reel at {at:.1f}s")
                continue
            label = f"{at:.1f}s"
            frames.append((label, output.read_bytes()))
            if placement is not None and shot is not None:
                placement[label] = {"shot": shot.index,
                                    "beat": shot.beat_index}

    if not frames and wanted:
        notes.append("no frame of the finished reel could be read back; "
                     "it ships ungraded")
    return frames, notes


# ---------------------------------------------------------------------------
# Music bed
#
# Tracks come from assets/music/ and nowhere else - nothing is downloaded, and
# nothing is generated. A track may carry a sidecar .txt holding the credit
# line its licence asks for; step 6 puts that in the post text, because a
# credit that only lives in a file next to the .mp4 is not a credit.
# ---------------------------------------------------------------------------

#: Ledger entries kept. Only the newest MUSIC_REUSE_COOLDOWN_RUNS matter for a
#: decision; the rest is history worth being able to read.
MUSIC_HISTORY_LIMIT = 200


def _read_json_file(path: Path, default: dict) -> dict:
    """Read a small JSON file, treating a corrupt one as absent."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(default)
    return data if isinstance(data, dict) else dict(default)


def _write_json_file(path: Path, data: dict) -> None:
    """Write a small JSON file atomically, so a crash cannot corrupt it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".part")
    try:
        partial.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                           encoding="utf-8")
        os.replace(partial, path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


@dataclass
class Track:
    """One music bed, and whatever its licence wants said about it."""

    path: Path
    credit: str = ""
    credit_required: bool = False
    duration: float = 0.0
    bpm: float = 0.0
    beat_offset: float = 0.0
    #: Somebody measured this track and found no constant grid in it. Not the
    #: same as bpm 0.0, which means nobody has looked.
    beatless: bool = False

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def has_grid(self) -> bool:
        """Whether cuts could actually be snapped to this bed.

        Asked *before* `resolve_track` applies the MUSIC_BPM fallback, so it
        has to account for it: a track with no tempo of its own still gets a
        grid where MUSIC_BPM stands one up. `beatless` overrides both - it is
        the measured answer that there is no constant pulse to snap to, and no
        fallback number can conjure one.

        A tempo outside MUSIC_BPM_MIN..MAX is a typo that `prepare_track` will
        drop later, so it does not count here either. Otherwise a stray
        "bpm: 9000" would pull the rotation towards the one bed guaranteed to
        end up with no grid at all.
        """
        if self.beatless:
            return False
        tempo = self.bpm or config.MUSIC_BPM
        return bool(tempo) and config.MUSIC_BPM_MIN <= tempo <= config.MUSIC_BPM_MAX

    def to_dict(self) -> dict:
        return {
            "file": self.name,
            "path": str(self.path.resolve()),
            "duration": self.duration,
            "credit": self.credit,
            "credit_required": self.credit_required,
            "loudness_target": config.MUSIC_LOUDNESS_TARGET,
            "ducking": bool(config.MUSIC_DUCKING),
            "bpm": self.bpm,
            "beat_offset": self.beat_offset,
            "beatless": self.beatless,
        }


#: A sidecar line that is a setting rather than a credit. Matched against the
#: line with its whitespace collapsed, so "bpm: 92", "bpm:92" and "BPM = 92"
#: are all the same thing and a tab does not defeat it.
_SIDECAR_META_RE = re.compile(
    r"^(bpm|beat_offset|offset) ?[:=] ?([0-9]+(?:[.][0-9]+)?)$", re.I
)

#: "bpm: none" - measured, and there is no constant grid in this track to snap
#: to. Distinct from *no* bpm line, which means nobody has looked yet, and the
#: distinction is the point: an ambient bed with no pulse cannot be fixed by
#: writing a number, so the run should say the bed is beatless once rather than
#: printing "add a bpm line" every day forever about a file that will never
#: have one. Measured on assets/music/: fitting a constant grid to the tracked
#: beats gives a residual of 7-14ms on the three beds that have a tempo and
#: 39-2018ms on the two that do not, against 110ms for pure noise.
_SIDECAR_BEATLESS_RE = re.compile(r"^(bpm|beat_offset|offset) ?[:=] ?none$", re.I)


def _read_sidecar(path: Path) -> tuple[str, dict[str, float]]:
    """Read `track.txt` beside `track.mp3`, if it is there.

    Returns the credit line and whatever settings were written under it. A
    sidecar is a credit by default and a settings file only where a line says
    so, which is what keeps every sidecar written before SHOT_SNAP_TO_BEAT
    existed reading exactly as it did.

    A `bpm:` line must never reach the credit, and that is the load-bearing
    half of this function: the credit is copied verbatim into attribution.txt
    and from there into the public post text by step 6, so a stray setting in
    it is published as part of somebody's name.
    """
    sidecar = path.with_suffix(".txt")
    if not sidecar.is_file():
        return "", {}
    try:
        text = sidecar.read_text(encoding="utf-8")
    except OSError:
        return "", {}

    credit: list[str] = []
    meta: dict[str, float] = {}
    for raw in text.splitlines():
        line = " ".join(raw.split())
        if not line:
            continue
        if _SIDECAR_BEATLESS_RE.match(line):
            # Kept out of the credit for the same reason a number is, and
            # recorded as a settings line rather than dropped: "somebody looked
            # and there is no grid" is the answer to a question the run would
            # otherwise ask again tomorrow.
            meta["beatless"] = 1.0
            continue
        found = _SIDECAR_META_RE.match(line)
        if not found:
            credit.append(line)
            continue
        key = found.group(1).lower()
        meta["bpm" if key == "bpm" else "beat_offset"] = float(found.group(2))
    return " ".join(credit), meta


def _music_credit(path: Path) -> tuple[str, bool]:
    """The credit line for a track, and whether it has to be used.

    Writing a credit down is taken as saying it has to be used: over-crediting
    costs a line of caption, under-crediting breaches a licence. Same .txt/.json
    sidecar idea step 2 uses for its scripts, one file simpler.
    """
    credit, _meta = _read_sidecar(path)
    return credit, bool(credit)


def _music_track(path: Path) -> Track:
    """A Track built from one file and its sidecar, settings and all."""
    credit, meta = _read_sidecar(path)
    return Track(
        path=path,
        credit=credit,
        credit_required=bool(credit),
        bpm=meta.get("bpm", 0.0),
        beat_offset=meta.get("beat_offset", 0.0),
        beatless=bool(meta.get("beatless")),
    )


def available_tracks() -> list[Track]:
    """Every usable track in assets/music/, in a stable order."""
    directory = config.MUSIC_DIR
    if not directory.is_dir():
        return []
    wanted = {ext.lower() for ext in config.MUSIC_EXTENSIONS}
    found = [
        entry for entry in sorted(directory.iterdir())
        if entry.is_file() and entry.suffix.lower() in wanted
    ]
    return [_music_track(entry) for entry in found]


class MusicLedger:
    """Which track backed which run, so a small library still rotates.

    With three tracks and no memory, the same one opens most reels: the
    directory listing is stable, so "the first track" is a constant. This makes
    the choice depend on what the last few runs used instead.
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or config.MUSIC_USAGE_FILE)
        self._data = _read_json_file(self.path, {"history": []})
        self._data.setdefault("history", [])

    def _recent(self) -> list[str]:
        """Track names most recently used first."""
        return [str(entry.get("track") or "")
                for entry in reversed(self._data["history"])]

    def choose(self, tracks: list[Track], *, cooldown: int,
               prefer_grid: bool = False) -> Track | None:
        """Least recently used track, preferring one outside the cooldown.

        `prefer_grid` puts a bed the cuts can be snapped to ahead of one they
        cannot, *within* whatever the cooldown has left free. Ordering it that
        way round rather than as an outright filter is what keeps the beatless
        beds in the rotation: with a cooldown of 3 and three gridded beds, the
        gridded ones are all blocked every fourth run and a beatless one comes
        up on its own. Measured on assets/music/, that is the difference
        between cutting to a grid on 3 runs in 5 and on 3 in 4.
        """
        if not tracks:
            return None
        recent = self._recent()

        def staleness(track: Track) -> tuple[int, str]:
            # Never used is as stale as it gets; otherwise the longest ago
            # wins. The name breaks ties, so one library always ranks the same.
            try:
                last = recent.index(track.name)
            except ValueError:
                last = len(recent)
            return (-last, track.name)

        ranked = sorted(tracks, key=staleness)
        if cooldown > 0:
            blocked = set(recent[:cooldown])
            fresh = [t for t in ranked if t.name not in blocked]
            # Every track inside the cooldown means the library is smaller than
            # the cooldown - rotate anyway rather than dropping the bed.
            if fresh:
                ranked = fresh
        if prefer_grid:
            gridded = [t for t in ranked if t.has_grid]
            if gridded:
                return gridded[0]
        return ranked[0]

    def record(self, run_id: str, track: Track) -> None:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        history = [entry for entry in self._data["history"]
                   if entry.get("run_id") != run_id]
        history.append({"run_id": run_id, "track": track.name, "at": stamp})
        self._data["history"] = history[-MUSIC_HISTORY_LIMIT:]
        _write_json_file(self.path, self._data)


def _named_track(name: str) -> Track | None:
    """Resolve a track named by path, or by bare filename inside MUSIC_DIR."""
    path = Path(name)
    if not path.is_absolute() and not path.exists():
        path = config.MUSIC_DIR / path
    if not path.is_file():
        return None
    return _music_track(path)


def resolve_track(music, warnings: list[str]) -> tuple[Track | None,
                                                       MusicLedger | None]:
    """Pick the bed for this reel.

    The `music` argument is a path to pin one track, False to go without, or
    None to let the rotation choose. Returns the track and the ledger to record
    it in - a pinned track has no ledger, because nothing is being rotated.

    A track named on the command line has to exist: it was typed just now, and
    silently ignoring it would hide the typo. MUSIC_FILE in .env is held to a
    softer standard - it was set once, months ago, and a moved file should cost
    a warning rather than every run from here on. An empty assets/music/ is not
    a problem at all: it is how a fresh clone arrives, and silence is a
    perfectly good bed.
    """
    if music is False or not config.MUSIC_ENABLED:
        return None, None

    if music not in (None, True):
        track = _named_track(str(music))
        if track is None:
            raise AssemblyError(f"music track not found: {music}")
        return track, None

    if config.MUSIC_FILE:
        track = _named_track(config.MUSIC_FILE)
        if track is not None:
            return track, None
        warnings.append(
            f"MUSIC_FILE points at {config.MUSIC_FILE!r}, which is not there - "
            f"falling back to whatever is in assets/music/"
        )

    tracks = available_tracks()
    if not tracks:
        return None, None       # nothing dropped in yet; not worth a warning

    ledger = MusicLedger()
    chosen = ledger.choose(
        tracks,
        cooldown=config.MUSIC_REUSE_COOLDOWN_RUNS,
        # Only worth preferring a grid where something is going to cut to it.
        # With snapping off the preference would trade away variety for a
        # property nothing reads.
        prefer_grid=config.MUSIC_PREFER_BEAT_GRID and config.SHOT_SNAP_TO_BEAT,
    )
    return chosen, ledger


def prepare_track(track: Track | None, warnings: list[str]) -> Track | None:
    """Probe the chosen track, dropping it if ffmpeg cannot read it.

    A broken bed must not cost the reel. Everything else in this module
    degrades rather than failing, and music is the least important input here.
    """
    if track is None:
        return None
    duration = probe_audio(track.path)
    if not duration:
        warnings.append(
            f"music track {track.name!r} could not be read and was left out "
            f"of the mix"
        )
        return None
    track.duration = duration

    # The tempo is settled here rather than where the grid is built, so the
    # track carries the one that was actually used and assembly.json records
    # it. A track states its own; MUSIC_BPM stands in for one that does not.
    if not track.bpm and config.MUSIC_BPM:
        track.bpm = config.MUSIC_BPM
        track.beat_offset = track.beat_offset or config.MUSIC_BEAT_OFFSET

    # A tempo is typed by hand into a text file, so it is checked like one. A
    # stray "bpm: 9000" would put a beat every 7ms, which is not a grid - every
    # cut would find one within a frame and SHOT_SNAP_TO_BEAT would silently
    # become a no-op that still claimed to have fired.
    if track.bpm and not (config.MUSIC_BPM_MIN <= track.bpm <= config.MUSIC_BPM_MAX):
        warnings.append(
            f"music track {track.name!r} states {track.bpm:g} BPM, outside "
            f"{config.MUSIC_BPM_MIN:g}-{config.MUSIC_BPM_MAX:g} - the beat grid "
            f"was left off for this reel"
        )
        track.bpm = 0.0

    # Say so when the feature is switched on and cannot run. `main` prints this
    # in its summary, but a scheduled run goes through run.py, which shows the
    # warnings and not the summary - so on the daily path SHOT_SNAP_TO_BEAT
    # looked enabled, cut nothing on the grid, and said nothing about it. Every
    # reel shipped so far reports 0 cuts on beat for exactly this reason: no
    # sidecar states a tempo and MUSIC_BPM is 0.
    if config.SHOT_SNAP_TO_BEAT and not track.bpm:
        if track.beatless:
            # A different sentence, because it is a different situation and
            # only one of them is a thing to go and do. This bed was measured
            # and has no constant grid; asking for a tempo line every day would
            # be asking for a number that cannot be measured off the audio.
            warnings.append(
                f"cutting on the beat is on, and {track.name!r} is marked "
                f"beatless - the cuts fall on the words instead"
            )
        else:
            warnings.append(
                f"cutting on the beat is on, but no tempo is known for "
                f"{track.name!r} - add a \"bpm: <n>\" line to "
                f"{Path(track.name).stem}.txt (or \"bpm: none\" if it has no "
                f"pulse, or set MUSIC_BPM) to switch it on"
            )
    return track


def beat_grid(track: Track | None, total: float) -> list[float]:
    """Every beat of the bed, in reel time. Empty when there is none to have.

    The track's clock and the reel's are the same one: `_music_chains` feeds
    the bed in with -stream_loop and trims it from 0, so the file's first
    sample is the reel's first frame and a beat is just `offset + n / bps`.
    That is the only reason this can be computed without decoding anything.

    `beat_offset` is where the first downbeat sits inside the file, for a track
    that does not open on one. It is folded back inside a single beat, so
    writing the second or the fiftieth downbeat down works as well as the first.

    The tempo is read off the track and nowhere else - `prepare_track` has
    already settled it, MUSIC_BPM fallback included - so what this builds a
    grid from is exactly what assembly.json says it did.
    """
    if track is None or total <= 0:
        return []
    bpm = track.bpm
    if bpm <= 0 or not (config.MUSIC_BPM_MIN <= bpm <= config.MUSIC_BPM_MAX):
        return []

    step = 60.0 / bpm
    offset = track.beat_offset % step
    count = int((total - offset) / step) + 1
    return [round(offset + index * step, 4) for index in range(max(0, count))]


def _on_grid(beats: list[float], at: float) -> bool:
    """Did this cut land on a beat, measured back against the grid?"""
    return any(abs(beat - at) <= _BEAT_TOLERANCE_SECONDS for beat in beats)


# ---------------------------------------------------------------------------
# Sound design
#
# One-shots come from assets/sfx/ and nowhere else - nothing is downloaded and
# nothing is generated, exactly like the music bed. The role is read off the
# front of the filename, so whoosh.wav, whoosh-02.wav and whoosh_soft.mp3 are
# all whooshes and the placement rotates between them.
#
# A whoosh across a cut is the cheapest production value in the whole pipeline:
# it costs one small file and it turns a change of picture into a deliberate
# edit. It is also the easiest thing here to overdo, which is what
# SFX_MIN_GAP_SECONDS exists to prevent.
# ---------------------------------------------------------------------------

#: Roles this module knows how to place. Anything else in assets/sfx/ is
#: reported rather than ignored - a file named wooosh.wav is a typo, and
#: silently doing nothing with it is how that typo survives for months.
SFX_ROLES = ("whoosh", "impact")

#: Peak measured below this is a file with no signal in it - a silent export,
#: or a lead-in so long that nothing lands where it was placed.
SFX_SILENCE_FLOOR_DB = -60.0

#: Bounds on the correction a file may be given. A near-silent one-shot must
#: not be lifted 70 dB into its own noise floor to reach the target.
SFX_MAX_GAIN_DB = 20.0
SFX_MIN_GAIN_DB = -40.0

_MAX_VOLUME_RE = re.compile(r"max_volume:\s*(-?\d+(?:\.\d+)?)\s*dB")


@dataclass
class Sfx:
    """One effect file, and what it has to be scaled by to sit in the mix."""

    role: str
    path: Path
    peak_db: float
    gain_db: float

    @property
    def name(self) -> str:
        return self.path.name


@dataclass
class SfxCue:
    """One effect, placed at one moment on the timeline."""

    sfx: Sfx
    at: float
    reason: str

    def to_dict(self) -> dict:
        return {
            "role": self.sfx.role,
            "file": self.sfx.name,
            "at": round(self.at, 3),
            "gain_db": round(self.sfx.gain_db, 2),
            "reason": self.reason,
        }


def _sfx_role(path: Path) -> str:
    """The role a filename declares: the letters before the first separator."""
    match = re.match(r"[a-z]+", path.stem.strip().lower())
    return match.group(0) if match else ""


def probe_peak_db(path: Path) -> float | None:
    """Peak level of an audio file in dBFS, or None if it cannot be measured.

    volumedetect rather than ebur128 on purpose. These are 200ms transients and
    EBU R128 integrates over 400ms blocks, so it would hand back nothing usable
    for exactly the files this is for. Peak is also what a sound editor levels
    a one-shot by, which makes it the right measure and not merely the
    available one.
    """
    args = [ffmpeg_binary(), "-hide_banner", "-nostdin", "-i", str(path),
            "-af", "volumedetect", "-f", "null", "-"]
    try:
        done = subprocess.run(args, capture_output=True, text=True,
                              errors="replace", timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None

    match = _MAX_VOLUME_RE.search(done.stderr or "")
    return float(match.group(1)) if match else None


def available_sfx() -> dict[str, list[Path]]:
    """Every one-shot in assets/sfx/, grouped by the role its name declares."""
    library: dict[str, list[Path]] = {}
    if not config.SFX_DIR.is_dir():
        return library

    suffixes = {s.lower() if s.startswith(".") else f".{s.lower()}"
                for s in config.SFX_EXTENSIONS}
    for path in sorted(config.SFX_DIR.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        library.setdefault(_sfx_role(path), []).append(path)
    return library


def resolve_sfx(sfx, warnings: list[str]) -> dict[str, list[Sfx]]:
    """Measure the one-shots this reel can use, keyed by role.

    `sfx` is False to go without and None to use whatever is in assets/sfx/.
    Everything degrades: an unreadable or silent file is dropped with a
    warning, an unknown role is reported, and an empty directory is not a
    problem at all - it is how a fresh clone arrives, and it means the reel
    sounds exactly as it did before any of this existed.
    """
    if sfx is False or not config.SFX_ENABLED:
        return {}

    found = available_sfx()
    if not found:
        return {}

    unknown = sorted(set(found) - set(SFX_ROLES))
    if unknown:
        warnings.append(
            "assets/sfx/ holds file(s) with no role this step places: "
            + ", ".join(f"{role or '(unnamed)'}*" for role in unknown)
            + f". Rename them to one of {', '.join(SFX_ROLES)} to use them."
        )

    library: dict[str, list[Sfx]] = {}
    for role in SFX_ROLES:
        for path in found.get(role, []):
            peak = probe_peak_db(path)
            if peak is None:
                warnings.append(
                    f"sound effect {path.name!r} could not be measured and was "
                    f"left out"
                )
                continue
            if peak <= SFX_SILENCE_FLOOR_DB:
                warnings.append(
                    f"sound effect {path.name!r} peaks at {peak:.1f} dBFS - "
                    f"there is no signal in it, so it was left out"
                )
                continue
            gain = max(SFX_MIN_GAIN_DB,
                       min(SFX_MAX_GAIN_DB, config.SFX_PEAK_TARGET - peak))
            library.setdefault(role, []).append(
                Sfx(role=role, path=path, peak_db=peak, gain_db=round(gain, 2))
            )
    return library


def plan_sfx(shots: list[Shot], total: float,
             library: dict[str, list[Sfx]]) -> list[SfxCue]:
    """Decide where the one-shots land.

    An impact on the first frame - the one effect with no cut to ride, which is
    the point of it - and a whoosh across each cut after that, thinned so two
    never crowd each other. The whoosh is placed SFX_LEAD_SECONDS *before* its
    cut so the transient lands on the change of picture rather than after it: a
    whoosh that arrives once the picture has already changed is heard as a
    separate noise instead of as the cut itself.
    """
    cues: list[SfxCue] = []
    placed_at: float | None = None

    impacts = library.get("impact") or []
    if config.SFX_IMPACT_AT_OPEN and impacts:
        cues.append(SfxCue(sfx=impacts[0], at=0.0, reason="the open"))
        placed_at = 0.0

    whooshes = library.get("whoosh") or []
    if config.SFX_ON_CUTS and whooshes:
        for shot in shots[1:]:          # every shot after the first opens on a cut
            if placed_at is not None and (
                shot.start - placed_at < config.SFX_MIN_GAP_SECONDS
            ):
                continue
            at = max(0.0, shot.start - config.SFX_LEAD_SECONDS)
            if at >= total:
                break
            chosen = whooshes[len(cues) % len(whooshes)]
            cues.append(SfxCue(sfx=chosen, at=at, reason=f"cut into shot {shot.index}"))
            placed_at = shot.start

    cues.sort(key=lambda cue: cue.at)
    return cues


def _sfx_chains(cues: list[SfxCue], first_index: int) -> tuple[list[str], list[str]]:
    """One filter chain per placed effect, and the labels they come out on."""
    chains: list[str] = []
    labels: list[str] = []
    for position, cue in enumerate(cues):
        label = f"sfx{position}"
        steps = [f"aresample={config.AUDIO_SAMPLE_RATE}"]
        if abs(cue.sfx.gain_db) >= 0.05:
            steps.append(f"volume={cue.sfx.gain_db:.2f}dB")
        delay = int(round(cue.at * 1000))
        if delay > 0:
            # all=1 or only the first channel is delayed and the effect arrives
            # twice, once in each ear.
            steps.append(f"adelay={delay}:all=1")
        chains.append(f"[{first_index + position}:a]" + ",".join(steps) + f"[{label}]")
        labels.append(label)
    return chains, labels

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


def music_credit_entry(track: Track, seconds: float) -> dict:
    """The bed, in the same shape as a footage credit.

    Deliberately the same shape, and deliberately in the same list: step 6
    reads `attributions` out of assembly.json without caring what kind of asset
    an entry describes, so a track that has to be credited reaches the post
    text by itself. `provider` stays empty because a local file has none - the
    courtesy "via Pexels" line skips blanks, which is what a track dropped in
    by hand should do.
    """
    return {
        "asset_index": None,
        "beat_index": None,
        "file": track.name,
        "kind": "music",
        "provider": "",
        "provider_id": None,
        "creator": None,
        "creator_url": None,
        "license": None,
        "source_url": None,
        "attribution": track.credit,
        "attribution_required": bool(track.credit_required),
        "screen_seconds": round(seconds, 3),
    }


def write_attribution_file(path: Path, manifest: dict, credits: list[dict]) -> None:
    """Human-readable credits next to the .mp4, for whoever posts it."""
    footage = [c for c in credits if c.get("kind") != "music"]
    music = [c for c in credits if c.get("kind") == "music"]
    required = [c for c in credits if c.get("attribution_required")]
    lines = [
        f"Attribution for: {manifest.get('topic') or '(untitled)'}",
        f"Run: {manifest.get('run_id', '?')}",
        f"Assembled: {datetime.now().astimezone().isoformat(timespec='seconds')}",
        "",
        f"{len(footage)} asset(s) appear in this video."
        + (f" {len(required)} require credit when posting." if required
           else " None require credit, but crediting is good manners."),
        "",
    ]
    for credit in footage:
        flag = "  [CREDIT REQUIRED]" if credit.get("attribution_required") else ""
        lines.append(
            f"- {credit['attribution'] or credit['file']}"
            f" ({credit['screen_seconds']}s on screen){flag}"
        )
    for credit in music:
        flag = ("  [CREDIT REQUIRED]" if credit.get("attribution_required")
                else "  (no credit line on file - check the licence yourself)")
        lines += ["", "Music:",
                  f"- {credit['attribution'] or credit['file']}{flag}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


#: What step 4 knew about an asset that is worth carrying onto the timeline.
#: Deliberately short: this is here to be correlated against how a reel did,
#: not to duplicate the visuals manifest, which is on disk and pointed at by
#: `visuals_manifest`. Provider is the one nobody could answer before - every
#: shot in the first eight reels reads `"provider": null` when the assembly is
#: loaded back, so "do the Pexels reels hold viewers better than the Openverse
#: ones?" had no way of being asked.
_SHOT_PROVENANCE_FIELDS = ("provider", "provider_key", "score", "relevance",
                           "width", "height")


def _shot_provenance(manifest: dict, shot: Shot) -> dict:
    """Step 4's record of the asset behind one shot, or nothing if it is gone.

    Missing keys are left out rather than written as null: a manifest from
    before a field existed should read as "not recorded", which an absent key
    says and a null does not.
    """
    assets = manifest.get("assets") or []
    if not 0 <= shot.asset_index < len(assets):
        return {}
    asset = assets[shot.asset_index]
    return {field: asset[field]
            for field in _SHOT_PROVENANCE_FIELDS if field in asset}


def _derive_hook_card(script: str) -> str:
    """Step 2's hook card, worked out here when the manifest has none.

    A run built before the hook pass existed, or one whose script came from
    somewhere other than step 2, still gets a card - cut from the opening
    sentence by the same two functions step 2 would have used, so the wording
    does not depend on which path it arrived by.
    """
    try:
        from pipeline import script_gen
    except Exception:      # noqa: BLE001 - a card is never worth a failed reel
        return ""
    opening, _ = script_gen.split_first_sentence(script or "")
    return script_gen.hook_card_text(opening)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def assemble_video(audio, visuals, *, output_dir: str | Path | None = None,
                   run_id: str | None = None, captions: bool | None = None,
                   alignment: str | None = None, music=None, sfx=None,
                   motion: str | None = None,
                   hook=None, cover=None, dry_run: bool = False,
                   overwrite: bool = True) -> dict:
    """Cut step 3's narration and step 4's footage into a finished reel.

    Args:
        audio: step 3's result dict, or a path to the narration .mp3.
        visuals: step 4's manifest dict, its manifest.json, or the run directory.
        output_dir: where to write. Defaults to output/<date>/<run id>/.
        run_id: overrides the manifest's run id, which names that directory.
        captions: burn in captions. Defaults to config.CAPTIONS_ENABLED.
        alignment: caption timing backend. Defaults to config.CAPTION_ALIGNMENT.
        music: a track to pin under the reel, or False for none. Defaults to
            rotating over whatever is in assets/music/.
        sfx: False to render without sound effects. Defaults to using whatever
            is in assets/sfx/.
        motion: how shots move - "alternate", "push" or "off". Defaults to
            config.SHOT_MOTION. Overridden per reel so step 9 can trial it.
        hook: text for the hook card burned over the opening. False for no
            card; None takes step 2's `hook_card` from the manifest, falling
            back to the script's own opening sentence.
        cover: where the cover frame comes from - "auto", "hook_card" or
            "footage". False writes none; None takes config.COVER_SOURCE.
        dry_run: plan the edit and write the captions, but do not encode.
        overwrite: replace an existing .mp4 at the destination.

    Returns:
        The assembly manifest, also written to assembly.json in the output
        directory: the video's path and length, the shot-by-shot timeline, the
        caption files, the cover frame and why it was chosen, and step 4's
        attribution for every asset used.

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
    # --- word timings, ahead of the edit -----------------------------------
    # This used to sit after the timeline was planned, because the only thing
    # that wanted it was the captions. Snapping cuts onto word boundaries needs
    # it *before*: it is what decides where the cuts go, and that cannot be
    # worked out after they have been placed.
    #
    # Which means a run with captions off but SHOT_SNAP_TO_WORDS on still pays
    # for one alignment. That is the deal being struck - cutting on the words
    # requires knowing where the words are - and SHOT_SNAP_TO_WORDS=false opts
    # out of both the cost and the feature.
    words: list[dict] = []
    backend: str | None = None
    loss: float | None = None
    if use_captions or config.SHOT_SNAP_TO_WORDS:
        words, backend, loss, align_warnings = align_words(
            audio_path, manifest, narration, backend_mode
        )
        warnings += align_warnings

    # --- the music bed, ahead of the edit ----------------------------------
    # Resolved before the timeline for the same reason the word timings are:
    # SHOT_SNAP_TO_BEAT puts the opening cuts on the bed's beat grid, and which
    # track is playing is what decides where those beats fall. Nothing is
    # written to the ledger here - that still waits for a finished encode.
    track, ledger = resolve_track(music, warnings)
    track = prepare_track(track, warnings)
    beats = beat_grid(track, narration + max(0.0, config.VIDEO_TAIL_SECONDS))

    shots, plan_warnings = plan_timeline(manifest, narration, probes, words,
                                         beats, motion)
    warnings += plan_warnings
    total = round(shots[-1].end, 3)

    # --- sound design ------------------------------------------------------
    sfx_library = resolve_sfx(sfx, warnings)
    sfx_cues = plan_sfx(shots, total, sfx_library) if sfx_library else []

    # --- the hook card -----------------------------------------------------
    if hook is False or not config.HOOK_CARD_ENABLED:
        card, card_source = "", "off"
    elif hook:
        card, card_source = " ".join(str(hook).split()), "given"
    elif manifest.get("hook_card"):
        card, card_source = str(manifest["hook_card"]).strip(), "step 2"
    else:
        card, card_source = _derive_hook_card(manifest.get("script") or ""), "script"
    card_seconds = min(config.HOOK_CARD_SECONDS, total) if card else 0.0

    # --- captions ----------------------------------------------------------
    caption_report = {
        "enabled": bool(use_captions), "backend": None, "cues": 0, "words": 0,
        "loss": None, "font": None, "style": config.CAPTION_STYLE,
        "ass_path": None, "srt_path": None, "alignment_path": None,
        "pop": (f"{config.CAPTION_POP_SCALE}% over {config.CAPTION_POP_MS}ms"
                if config.CAPTION_POP else None),
        "payoff_words": 0,
    }
    ass_path: Path | None = None
    fonts_dir: Path | None = None
    # Resolved below only when there is text to draw. The cover reads them too,
    # and only ever when it is drawing the card - which is the case that fills
    # them in.
    font_name: str = config.CAPTION_FONT_NAME
    font_file: Path | None = None

    # A card with the captions switched off still needs a subtitle file - it is
    # drawn by the same libass pass - so the two are decided together.
    if use_captions or card:
        cues = group_cues(words, total) if use_captions else []
        font_name, font_file, fonts_dir = resolve_font()
        if font_file is None:
            warnings.append(
                f"no font file found for {font_name!r} - libass will substitute "
                f"if it cannot resolve the name either. Drop a .ttf into "
                f"assets/fonts/ to pin the look."
            )

        ass_path = destination / CAPTIONS_ASS_NAME
        ass_path.write_text(
            build_ass(cues, font_name, font_file,
                      hook=card, hook_seconds=card_seconds),
            encoding="utf-8",
        )
        caption_report["ass_path"] = str(ass_path.resolve())
        caption_report["font"] = font_name

    if use_captions:
        srt_path = destination / CAPTIONS_SRT_NAME
        srt_path.write_text(build_srt(cues), encoding="utf-8")
        alignment_path = destination / ALIGNMENT_NAME
        alignment_path.write_text(
            json.dumps({"backend": backend, "loss": loss, "words": words},
                       indent=2, ensure_ascii=False), encoding="utf-8"
        )
        caption_report.update({
            "backend": backend, "cues": len(cues), "words": len(words),
            "loss": round(loss, 4) if loss else loss,
            "srt_path": str(srt_path.resolve()),
            "alignment_path": str(alignment_path.resolve()),
            # Counted rather than styled-and-forgotten: how often a reel has a
            # number to make is the only thing that says whether the payoff
            # colour is doing anything, and step 9 can only correlate what got
            # written down.
            "payoff_words": sum(1 for cue in cues for word in cue.words
                                if _is_payoff_word(word["text"])),
        })

    # --- the cover frame ---------------------------------------------------
    # Planned here and written after the encode. A dry run gets the whole
    # decision - which shot, which frame, why - and no file, the same way it
    # gets a timeline and no .mp4.
    #
    # Step 5b goes first, because after the choice there is nothing left to
    # choose: the candidates are rendered at look size and graded, and those
    # verdicts are what `cover_score` weighs against its proxies. Skipped on a
    # dry run - it costs an encoder pass and a model call, and a dry run is
    # meant to cost neither.
    cover_looks: dict[str, dict] = {}
    if (cover is not False and config.COVER_ENABLED and shots and not dry_run
            and config.LOOK_ENABLED and config.COVER_LOOK_ENABLED):
        shortlist = cover_candidates(
            shots, card=card, card_seconds=card_seconds,
            font_file=font_file, source=cover,
        )
        previews, notes = preview_cover_frames(
            shortlist["pool"], card=card, card_seconds=card_seconds,
            want_card=shortlist["want_card"], font_name=font_name,
            font_file=font_file, fonts_dir=fonts_dir,
        )
        warnings += notes
        cover_looks, notes = look.judge_covers(
            previews, topic=manifest.get("topic") or "", hook=card,
        )
        warnings += notes

    cover = None if cover is False else plan_cover(
        shots, manifest, probes, card=card, card_seconds=card_seconds,
        font_file=font_file, source=cover, looks=cover_looks,
        warnings=warnings,
    )

    # --- render ------------------------------------------------------------
    reel_look = look.blank_grade()
    if dry_run:
        rendered = {"duration": total, "ok": False}
    else:
        warnings += render(
            shots, audio_path, video_path,
            total=total, narration=narration,
            ass_path=ass_path, fonts_dir=fonts_dir,
            work_dir=destination, music=track, sfx=sfx_cues,
            overwrite=overwrite,
        )
        if track is not None and ledger is not None:
            # Recorded after the encode, not before: a run that never produced
            # a reel must not push a track into the cooldown.
            ledger.record(identifier, track)
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
        if cover is not None:
            warnings += write_cover(cover, shots, destination,
                                    font_name=font_name, font_file=font_file,
                                    fonts_dir=fonts_dir)

        # --- step 5b: grade what was rendered, not what was planned --------
        # Reads reel.mp4 back and changes nothing about it. The file is already
        # on disk, so this cannot cost a reel - what it costs is the honesty of
        # the row step 9 stores, which until now described the plan.
        if config.LOOK_ENABLED and config.REEL_LOOK_ENABLED:
            placement: dict[str, dict] = {}
            frames, notes = sample_reel_frames(
                video_path, float(rendered.get("duration") or total),
                shots=shots, placement=placement,
            )
            warnings += notes
            reel_look, notes = look.grade_reel(
                frames,
                topic=manifest.get("topic") or "",
                script=str(manifest.get("script") or ""),
                lines=narration_lines(manifest),
            )
            warnings += notes
            _place_verdicts(reel_look, placement)

    # --- carry step 4's attribution forward --------------------------------
    credits = collect_attribution(manifest, shots)
    if track is not None:
        credits.append(music_credit_entry(track, total))
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
        "music": track.to_dict() if track else None,
        "narration_duration": narration,
        "tail_seconds": round(total - narration, 3),
        "visuals_manifest": manifest.get("manifest_path"),
        "asset_dir": manifest.get("asset_dir"),
        "captions": caption_report,
        "hook_card": {
            "text": card,
            "seconds": round(card_seconds, 3),
            "source": card_source,
        },
        "cover": cover,
        # The only part of this file written by something that saw the video.
        # Everything else here - the timeline, the stats, the shot list - is a
        # record of what was asked for.
        "look": reel_look,
        "sfx": [cue.to_dict() for cue in sfx_cues],
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
                "motion": shot.motion,
                "motion_zoom": shot.motion_zoom,
                **_shot_provenance(manifest, shot),
            }
            for shot in shots
        ],
        "attributions": credits,
        "attribution_path": str(attribution_path.resolve()),
        "warnings": list(dict.fromkeys(warnings)),
        "stats": {
            "shots": len(shots),
            "beats": len({shot.beat_index for shot in shots}),
            "assets_used": sum(1 for c in credits if c.get("kind") != "music"),
            "assets_sourced": len(manifest["assets"]),
            # Carried up from the visuals manifest so step 9 can ask whether
            # a reel built out of its own neutral shot reaches worse. It was
            # counted there from the start and read by nothing.
            "beats_from_neutral": (
                (manifest.get("stats") or {}).get("beats_from_neutral")
            ),
            "looped_shots": sum(1 for shot in shots if shot.looped),
            "reused_shots": sum(1 for shot in shots if shot.reused),
            "moving_shots": sum(1 for shot in shots if shot.motion),
            "mean_shot_seconds": round(total / len(shots), 2) if shots else 0.0,
            "first_shot_seconds": shots[0].duration if shots else 0.0,
            "sfx_cues": len(sfx_cues),
            "cuts_on_beat": sum(1 for shot in shots[1:]
                                if _on_grid(beats, shot.start)),
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
          f"(first {stats.get('first_shot_seconds', 0):.2f}s, "
          f"{stats['assets_used']}/{stats['assets_sourced']} assets used)")
    card = assembly.get("hook_card") or {}
    if card.get("text"):
        print(f"HOOK    : \"{card['text']}\" for {card['seconds']:g}s "
              f"(from {card['source']})")
    if captions["enabled"]:
        payoffs = captions.get("payoff_words") or 0
        print(f"CAPTIONS: {captions['cues']} cues from {captions['words']} words "
              f"via {captions['backend']}"
              + (f" (loss {captions['loss']})" if captions["loss"] else "")
              + f", {captions['font']} {captions['style']}")
        if captions.get("pop") or payoffs:
            print(f"          pop {captions.get('pop') or 'off'}"
                  + (f", {payoffs} payoff word(s) in "
                     f"{config.CAPTION_PAYOFF_COLOR}" if payoffs else ""))
    else:
        print("CAPTIONS: off")
    music = assembly.get("music")
    if music:
        duck = ("ducked under the voice" if music["ducking"] else "flat level")
        print(f"MUSIC   : {music['file']} "
              f"@ {music['loudness_target']:g} LUFS, {duck}"
              + ("  [CREDIT REQUIRED]" if music["credit_required"] else ""))
        on_beat = stats.get("cuts_on_beat") or 0
        if music.get("bpm"):
            print(f"          {music['bpm']:g} BPM"
                  + (f" (first beat {music['beat_offset']:g}s)"
                     if music.get("beat_offset") else "")
                  + f", {on_beat} of {max(0, stats['shots'] - 1)} cuts on the grid")
        elif config.SHOT_SNAP_TO_BEAT:
            print(f"          no tempo - add \"bpm: <n>\" to "
                  f"{Path(music['file']).stem}.txt to cut on the beat")
    else:
        # Only suggest the fix when it is the problem - a run told --no-music
        # does not need to be advised to add music.
        hint = "" if available_tracks() else "  (drop a track into assets/music/)"
        print(f"MUSIC   : none{hint}")
    effects = assembly.get("sfx") or []
    if effects:
        by_role: dict[str, int] = {}
        for cue in effects:
            by_role[cue["role"]] = by_role.get(cue["role"], 0) + 1
        print("SFX     : " + ", ".join(f"{n}x {role}"
                                       for role, n in sorted(by_role.items()))
              + f" @ {config.SFX_PEAK_TARGET:g} dBFS peak")
    elif config.SFX_ENABLED and not available_sfx():
        print("SFX     : none  (drop whoosh.wav / impact.wav into assets/sfx/)")

    cover = assembly.get("cover")
    if cover:
        print(f"COVER   : {cover['source']} at {cover['time']:g}s "
              f"(shot {cover['shot_index']}, {cover['file'][:34]})"
              + ("" if cover["written"] else "  (dry run - not written)"))
        print(f"          {cover['reason']}")
        seen = cover.get("look") or {}
        if seen:
            others = [entry for entry in cover.get("look_candidates") or []
                      if entry.get("shot_index") != cover["shot_index"]
                      and entry.get("grade") is not None]
            against = (" (others "
                       + "/".join(str(entry["grade"]) for entry in others) + ")"
                       if others else "")
            print(f"          looked at: {seen.get('grade')}/{look.GRADE_MAX}"
                  + ("" if seen.get("ok", True) else " REJECTED")
                  + f"{against} - {seen.get('why') or 'no reason given'}")

    elif config.COVER_ENABLED:
        print("COVER   : none")

    graded = assembly.get("look") or {}
    if graded.get("graded"):
        problems = ", ".join(graded.get("problems") or []) or "nothing flagged"
        print(f"LOOK    : {graded['grade']}/{look.GRADE_MAX} over "
              f"{len(graded.get('frames') or [])} frames ({problems})")
        if graded.get("summary"):
            print(f"          {graded['summary']}")
    elif config.LOOK_ENABLED and config.REEL_LOOK_ENABLED and not assembly["dry_run"]:
        print("LOOK    : ungraded")

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
    parser.add_argument("--music", type=Path, default=None,
                        help="pin one background track (path, or a filename "
                             "inside assets/music/). Default: rotate over "
                             "whatever is in assets/music/")
    parser.add_argument("--no-music", action="store_true",
                        help="render without a music bed")
    parser.add_argument("--no-sfx", action="store_true",
                        help="render without the one-shots in assets/sfx/")
    parser.add_argument("--hook", default=None,
                        help="text for the hook card burned over the opening "
                             "(default: step 2's, or the script's first line)")
    parser.add_argument("--no-hook-card", action="store_true",
                        help="render without the hook card")
    parser.add_argument("--cover", default=None,
                        choices=("auto", "hook_card", "footage"),
                        help=f"where the cover frame comes from "
                             f"(default: {config.COVER_SOURCE})")
    parser.add_argument("--no-cover", action="store_true",
                        help="do not write a cover frame")
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
            music=False if args.no_music else args.music,
            sfx=False if args.no_sfx else None,
            hook=False if args.no_hook_card else args.hook,
            cover=False if args.no_cover else args.cover,
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
