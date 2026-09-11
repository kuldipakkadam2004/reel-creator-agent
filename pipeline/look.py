"""Step 5b - look at the frames, not at the words about them.

Every other judgement in this pipeline is made on text. Step 0 ranks a topic on
its wording, step 2 writes and rewrites prose, step 4 screens candidate footage
on the stock library's own description of it, and step 5 chooses a cover from
the clip's duration, aspect ratio and resolution. Not one of them had decoded a
frame, and three consequences followed from the same absence:

  - `assemble.cover_score` asks, in its own docstring, "whether one frame of
    it, with no motion and no sound, is worth a stranger's tap" - and answered
    with shot length, aspect ratio, resolution and whether the clip had been
    reused. Every term is a proxy. None is the question.
  - step 4's screener prompt told the model that framing, colour and production
    quality were "handled elsewhere and are not your concern". They were
    handled nowhere. The claim had been true of an intention, never of code.
  - the features row step 9 correlates against reach recorded shots, beats and
    mean shot length: a description of the plan. An insights loop regressing
    reach against what a run *meant* to make cannot see the thing that was
    actually posted.

This module is the one place a picture is put in front of the model. It knows
nothing about ffmpeg, shots or run directories: callers hand it JPEG bytes with
the label each frame should be called by, and it hands back verdicts. That
split is what keeps the frame-grabbing in `assemble.py`, which already owns an
encoder, and keeps this file about the question being asked.

Nothing here raises. A look is an improvement to a reel, not a step in making
one - the same rule step 4's screener follows - so every failure comes back as
a warning beside an empty verdict, and the caller carries on with the blind
score it already had.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# Allow both `python pipeline/look.py` and `python -m pipeline.look`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402

#: 0-3, and the same scale for a cover frame and for the reel as a whole, so
#: the two numbers in assembly.json mean the same thing. Spelled out in both
#: prompts rather than left to the model, because "rate this out of 3" without
#: a rubric is a vote on how generous the model feels.
GRADE_MAX = 3

#: The closed vocabulary the reel grade may name a problem from. Closed on
#: purpose: step 9 turns each of these into a facet, and a facet is a question
#: asked of every reel in the history. Free text would give a vocabulary of
#: one-off phrasings, each with a sample size of one, and `build_vocabulary`
#: would drop the lot - so an open list would read as "the model found nothing"
#: however much it found.
PROBLEMS: tuple[str, ...] = (
    "dark",                 # underexposed to the point of being hard to read
    "blurry",               # out of focus, or upscaled past what it has
    "watermark",            # a provider's stamp, a logo, a burned-in caption
    "static",               # nothing in the frame is moving or changing
    "off_topic",            # the footage is not about what is being said
    "caption_unreadable",   # the running caption cannot be read over this
    "repetitive",           # the same shot or setup keeps coming back
    "artifact",             # compression mush, banding, a stretched crop
)

_COVER_SYSTEM_INSTRUCTION = """\
You judge single still frames as thumbnails for short-form vertical video.

A thumbnail has one job and it is not the same job as the video's: it is shown \
small, in a grid, with no motion and no sound, to somebody who is scrolling \
past and has not decided to watch anything. It has to survive that. You are \
blunt, you are specific about what you can actually see in the frame, and you \
never credit a frame for what the video around it might be doing."""

_COVER_PROMPT_TEMPLATE = """\
Topic of the video: {topic}
Its opening line: {hook}

Above are {count} candidate frames for this video's cover, each labelled. They \
are rendered exactly as they would ship: the real crop, and where the video \
opens on a headline card, that card drawn over the picture.

For each one, answer the only question a cover has to answer - with no motion \
and no sound, is this frame worth a stranger's tap?

Grade each from 0 to {grade_max}:
  {grade_max} - would stop a scroll on its own. A clear subject, a striking or \
unusual image, something a person would want explained.
  2 - clear and attractive, and you can tell what it is about, but it would \
not stop anybody who was not already slowing down.
  1 - legible, and nothing more. A generic or flat picture of the subject.
  0 - should not be a cover. Too dark or too busy to read small, out of focus, \
mid-motion-blur, a provider's watermark or logo across it, an unrecognisable \
crop, or nothing identifiable in it at all.

Set "ok" to false where you would not ship the frame as a cover at all - which \
is every 0, and any frame at any grade carrying something disqualifying.

Where a headline card is drawn over the frame, judge the type as part of it: a \
card that cannot be read against what is behind it is a cover problem, not a \
typography problem, and it is the most common way a cover fails.

"why" is a short phrase - a few words - naming what you actually see. On a \
reject say what is wrong with it; on a {grade_max} say what would stop the \
scroll.

Return ONLY a JSON array, one object per frame, using each frame's own label:
[{{"frame": "A", "grade": 2, "ok": true, "why": "clear shot of a river mouth"}}]"""

_REEL_SYSTEM_INSTRUCTION = """\
You review finished short-form vertical video by looking at frames sampled \
from it.

You are the last check before a video is posted, and the only one in the whole \
process that sees the video rather than the plan for it. Everything before you \
read a script, a shot list, or a stock library's own words for a clip - so the \
failures that reach you are exactly the ones words do not show: footage that \
is dark, soft or watermarked, footage that does not depict what is being said, \
captions that cannot be read over what is behind them, and the same shot \
arriving over and over. You say what is there. You do not soften it, and you \
do not guess at what the frames between the ones you were given might fix."""

_REEL_PROMPT_TEMPLATE = """\
Topic of the video: {topic}

What is being said, in order:
{script}

Above are {count} frames sampled evenly across the finished video, in order \
and labelled with where in the video each one falls. The running caption is \
burned into the video, so the words across a frame are what the viewer sees.

Grade the video from 0 to {grade_max} on what these frames show:
  {grade_max} - the footage looks bought rather than scraped, it depicts what \
is being said, and every caption is readable over what is behind it.
  2 - solid. Nothing here would stop you posting it, whatever small things you \
would change.
  1 - you would not post this as it stands. Name what is wrong.
  0 - broken. The footage is unusable, unrelated to the narration, or the \
captions cannot be read.

List the problems you can actually see, using ONLY these names:
{problems}

Use a name only where a viewer would notice it, not where you can find a trace \
of it if you look. An empty list is a real answer and the right one for a video \
with nothing wrong.

Also grade each frame 0 to {grade_max} on the same scale, judged as one \
picture, and say in a short phrase what it shows.

"summary" is one sentence on the video as a whole, for somebody who will not \
see the frames.

Return ONLY a JSON object:
{{"grade": 2, "problems": ["dark"], "summary": "...", "frames": \
[{{"frame": "0.0s", "grade": 2, "shows": "a river delta from above"}}]}}"""


# ---------------------------------------------------------------------------
# Talking to the model
# ---------------------------------------------------------------------------

def _strip_fences(raw: str) -> str:
    """Drop a ```json fence. Asked for JSON, models still sometimes wrap it."""
    text = str(raw or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse_json(raw: str, opener: str, closer: str):
    """Read the first JSON value of the expected shape out of a reply."""
    text = _strip_fences(raw)
    start, end = text.find(opener), text.rfind(closer)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"no JSON {opener}{closer} in the reply: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def _clamp_grade(value) -> int:
    """A grade the rest of the code can rely on being inside the scale."""
    try:
        return max(0, min(GRADE_MAX, int(round(float(value)))))
    except (TypeError, ValueError):
        return 0


def _ask(prompt: str, *, system_instruction: str,
         frames: list[tuple[str, bytes]]) -> str:
    """One multimodal call, through the same door every other call uses.

    `script_gen.call_gemini` is described in its own docstring as the only
    place the SDK is spoken to, and that is worth more than a shortcut: the key
    lookup, the pro-model quirks, the retries and the backoff are decided once.
    It grew an `images` argument for this caller rather than this caller
    growing a second client, and a `timeout` one because a multimodal call is
    slower than a text one by about the time it takes to read the images.
    """
    from pipeline import script_gen

    return script_gen.call_gemini(
        prompt,
        system_instruction=system_instruction,
        temperature=0.0,
        max_output_tokens=4096,
        json_mode=True,
        images=frames,
        timeout=config.LOOK_TIMEOUT_SECONDS,
    )


def _key_missing(what: str) -> list[str]:
    """The one reason a look is skipped that is not a failure."""
    try:
        config.require(*config.REQUIRED_SCRIPT_GEN)
    except config.MissingConfigError as exc:
        return [f"{what} was skipped: {exc}"]
    return []


# ---------------------------------------------------------------------------
# The cover
# ---------------------------------------------------------------------------

def judge_covers(frames: list[tuple[str, bytes]], *, topic: str = "",
                 hook: str = "") -> tuple[dict[str, dict], list[str]]:
    """Grade candidate cover frames. Returns ({label: verdict}, warnings).

    A verdict is {"grade": 0-GRADE_MAX, "ok": bool, "why": str}. Labels come
    back exactly as they were handed in; a label the model invents or drops is
    ignored rather than guessed at, because a verdict matched to the wrong
    frame is worse than no verdict - it would move the cover choice with
    confidence and leave a plausible reason in assembly.json.

    An empty map means the look did not happen, and the caller keeps the blind
    ranking it already had.
    """
    if not frames:
        return {}, []
    if not (config.LOOK_ENABLED and config.COVER_LOOK_ENABLED):
        return {}, []

    missing = _key_missing("the cover look")
    if missing:
        return {}, missing

    prompt = _COVER_PROMPT_TEMPLATE.format(
        topic=topic or "(not given)",
        hook=hook or "(none)",
        count=len(frames),
        grade_max=GRADE_MAX,
    )
    try:
        raw = _ask(prompt, system_instruction=_COVER_SYSTEM_INSTRUCTION,
                   frames=frames)
        data = _parse_json(raw, "[", "]")
    except Exception as exc:  # noqa: BLE001 - any failure keeps the blind score
        return {}, [f"the cover look failed ({type(exc).__name__}: {exc}); the "
                    f"cover was chosen without seeing the frames"]

    if not isinstance(data, list):
        return {}, ["the cover look did not return a JSON array; the cover was "
                    "chosen without seeing the frames"]

    known = {label for label, _ in frames}
    verdicts: dict[str, dict] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("frame") or "").strip()
        if label not in known:
            continue
        verdicts[label] = {
            "grade": _clamp_grade(entry.get("grade")),
            "ok": bool(entry.get("ok", True)),
            "why": str(entry.get("why") or "").strip()[:120],
        }

    notes: list[str] = []
    unseen = sorted(known - set(verdicts))
    if unseen:
        # Not fatal and not silent: the shots it names keep their blind score
        # and compete against shots that were graded, which is a quietly unfair
        # race unless it is written down.
        notes.append("the cover look returned no verdict for " + ", ".join(unseen))
    return verdicts, notes


# ---------------------------------------------------------------------------
# The finished reel
# ---------------------------------------------------------------------------

def blank_grade() -> dict:
    """The record a reel carries when nothing looked at it.

    Its own function because three callers need to agree on it - the grader's
    early returns, `assemble` on a dry run, and the tests - and because
    `graded` being False is load-bearing rather than cosmetic.
    """
    return {"graded": False, "grade": None, "problems": [],
            "summary": "", "frames": []}


def grade_reel(frames: list[tuple[str, bytes]], *, topic: str = "",
               script: str = "") -> tuple[dict, list[str]]:
    """Grade the rendered reel from frames sampled across it.

    Returns (record, warnings). The record is what goes into assembly.json and,
    through it, into the features row step 9 reads:

        {"graded": bool, "grade": 0-GRADE_MAX, "problems": [...],
         "summary": str, "frames": [{"frame", "grade", "shows"}, ...]}

    `graded` is the column that matters and the reason the rest are not enough
    on their own: a reel that was never looked at and a reel that was looked at
    and found clean both have no problems, and scoring them together is exactly
    the mistake `_written_since` exists to stop step 9 making.
    """
    record = blank_grade()
    if not frames:
        return record, []
    if not (config.LOOK_ENABLED and config.REEL_LOOK_ENABLED):
        return record, []

    missing = _key_missing("the reel look")
    if missing:
        return record, missing

    prompt = _REEL_PROMPT_TEMPLATE.format(
        topic=topic or "(not given)",
        script=(script or "(not given)").strip(),
        count=len(frames),
        grade_max=GRADE_MAX,
        problems="  " + "\n  ".join(PROBLEMS),
    )
    try:
        raw = _ask(prompt, system_instruction=_REEL_SYSTEM_INSTRUCTION,
                   frames=frames)
        data = _parse_json(raw, "{", "}")
    except Exception as exc:  # noqa: BLE001 - a look never costs a reel
        return record, [f"the reel look failed ({type(exc).__name__}: {exc}); "
                        f"this reel ships ungraded"]

    if not isinstance(data, dict):
        return record, ["the reel look did not return a JSON object; this reel "
                        "ships ungraded"]

    known = {label for label, _ in frames}
    per_frame = []
    for entry in data.get("frames") or []:
        if not isinstance(entry, dict):
            continue
        label = str(entry.get("frame") or "").strip()
        if label not in known:
            continue
        per_frame.append({
            "frame": label,
            "grade": _clamp_grade(entry.get("grade")),
            "shows": str(entry.get("shows") or "").strip()[:120],
        })

    # Only names from the closed list survive. A model that answers "slightly
    # washed out" has said something true and unusable: step 9 would carry it
    # as a facet of one reel forever.
    problems: set[str] = set()
    dropped: list[str] = []
    for name in (data.get("problems") or []):
        tidied = str(name).strip().lower().replace(" ", "_")
        if tidied in PROBLEMS:
            problems.add(tidied)
        elif str(name).strip():
            # Reported in the model's own words, not the tidied form. The note
            # exists so a name worth adding to PROBLEMS can be recognised, and
            # "slightly_washed_out" is harder to recognise than what was said.
            dropped.append(str(name).strip())

    record.update({
        "graded": True,
        "grade": _clamp_grade(data.get("grade")),
        "problems": sorted(problems),
        "summary": str(data.get("summary") or "").strip()[:300],
        "frames": per_frame,
    })

    notes: list[str] = []
    if dropped:
        notes.append("the reel look named problems outside its list and they "
                     "were dropped: " + ", ".join(dropped[:5]))
    if record["grade"] <= config.REEL_LOOK_WARN_AT:
        detail = ", ".join(sorted(problems)) or "no specific problem named"
        notes.append(f"the reel look graded this {record['grade']}/{GRADE_MAX} "
                     f"({detail}): {record['summary'] or 'no summary given'}")
    return record, notes
