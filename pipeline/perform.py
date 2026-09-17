"""Step 9 - how the reel actually did, and what that changes.

Every other step in this pipeline optimises a proxy. Step 0 scores velocity,
curiosity and footage; step 2 scores hooks; step 6 bands hashtags by post
count. Not one of those numbers has ever been checked against a view. This is
the step that closes that loop, and it is deliberately the last one built: a
feedback loop over a pipeline still changing shape would have been learning
about the pipeline rather than about the reels.

The work is small, because of something already true everywhere else - every
scoring decision in this repo emits the parts it was made of, not just a total:

    "hook": {"score": 2.3, "components": {"length": 1.5, "strong opener": 0.8}}
    parts["curiosity (contradiction, scale)"] = 1.0

So a finished reel is already a fully labelled example missing exactly one
column - what happened when it was posted. Two files supply it:

    output/<date>/<run>/features.json   what this reel was, written at run time
    output/performance.json             what it did, written by you afterwards

Three things are worth saying, because each is a place the obvious version is
wrong:

* **the components are stored, the scores are not.** Retune EXPLAINER_BONUS and
  every score in the history becomes uninterpretable, while "explainer shape
  fired on this reel" stays true forever. A row is a list of names; the
  arithmetic that turned them into 4.42 is not preserved and should not be;
* **a reel you decided not to post is data, and it arrives first.** It is a
  quality judgement you make the same day, for free, where an Instagram number
  takes 48 hours and a Meta app. `--skipped` records it, and for the first
  month it is the only signal that exists;
* **views trend with follower count.** Reel #40 beats reel #5 for reasons that
  have nothing to do with its hook, so nothing here compares raw numbers. Every
  reel is scored as a ratio to the median of its own neighbours in time, which
  takes the growth curve out. Watch-through is the default metric for the same
  reason: it is the one that does not simply climb with the account.

What it will not do is retune the weights by itself. Nothing here writes to
config, and `PERFORMANCE_FEEDBACK` is off until there are enough reels to argue
with. Read the report, move the constants yourself - a board that prints the
case for its own ranking is the property that makes step 0 trustworthy, and an
optimiser quietly editing those numbers would cost exactly that.

Standalone use:
    python pipeline/perform.py                    # which reels are waiting for numbers
    python pipeline/perform.py cd1bccae --views 4200 --watched 61 --saves 30
    python pipeline/perform.py cd1bccae --skipped "footage was of the simile"
    python pipeline/perform.py --report           # what the ledger has learned
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import random
import statistics
import sys
from datetime import date, datetime, timedelta, timezone
from importlib import import_module
from pathlib import Path

# Allow both `python pipeline/perform.py` and `python -m pipeline.perform`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from pipeline import visuals  # noqa: E402

#: Bumped when the shape of a features row changes. Old rows are kept and read
#: as best they can be - a ledger that discarded its history on an upgrade
#: would throw away the one thing here that takes months to collect.
#: 2 adds the structure pass's columns and the count of shots that move. A
#: version 1 row carries neither, and `facets` declines to ask it those
#: questions rather than reading the absence as "no components fired" or "the
#: footage was static" - the same distinction a backfilled row already draws.
#: 3 is the first version whose `music` column is true. Step 5 writes the
#: track's filename under "file"; this read it as "name" and got None every
#: time, so every row up to and including version 2 says the reel had no music
#: whether it had a bed or not. They are not evidence of silence, they are
#: evidence of a typo, and `facets` skips the question for them rather than
#: scoring every reel ever made as music-off.
#: 4 adds the take-away pass's columns. That pass exists because saves and
#: shares are 0 across every posted reel, so these are the first columns here
#: aimed at those two metrics rather than at watch-through - and the first that
#: were added before there was anything to correlate them against, deliberately
#: so that the correlation is possible later. A version 3 row is not "handed
#: nothing over", it is a reel made before anything asked.
#: 5 adds `cover_source` and `hashtag_names`, both of which were deferred until
#: there were numbers to correlate against and now are not. `hashtag_names` is
#: the one that changes what can be asked: every row up to here stored the
#: hashtag *count* and a tally of bands, so "did #woodwideweb do anything" was
#: not a question the ledger could hold, whatever the tags were. The names make
#: each tag a facet like any other. An older row has neither column and is not
#: asked either question - "no cover" and "no such column" are not the same.
#: 6 adds step 5b's columns, and they are the first in this row that describe
#: the reel rather than the plan for it. Everything above them - shots, beats,
#: mean shot length, the cover's source, which components fired - is a record
#: of what the run asked for. Nothing here had ever been read off the finished
#: file, because until step 5b nothing in the pipeline decoded a frame, so the
#: insights loop was regressing reach against a description of an intention.
#: `look_graded` is the load-bearing one: a reel nobody looked at and a reel
#: that was looked at and found clean both carry an empty `look_problems`, and
#: scoring them together would put the whole back catalogue in the clean arm.
#: 7 adds `beats_from_neutral`, which step 4 had counted into its manifest from
#: the start and nothing had ever read. It is the share of the reel given over
#: to the one safe shot the query writer picks for the whole video, and it is
#: the only column that separates the two reels step 5b graded 1/3 from the one
#: it graded 2. An older row cannot answer it: the count exists in manifests
#: going back only as far as the counter, and "no such column" is not "none".
FEATURES_VERSION = 7

#: 2 adds `watch_ratio` and the `measurements` history, and both are derived
#: for rows already in the file rather than left blank - the numbers needed
#: were always there, and a migration that only helped future reels would have
#: nothing to report for months.
LEDGER_VERSION = 2

FEATURES_FILENAME = "features.json"

#: What a reel can be. Every reel is registered `pending` when it is made: it
#: exists, and nobody has said yet whether it went out.
STATUSES = ("pending", "posted", "skipped")

#: Metrics the ledger stores. Counts are integers, `watch_through` is a
#: fraction, `watch_ratio` is that fraction left uncapped, `avg_watch_seconds`
#: is seconds.
METRICS = ("views", "reach", "watch_through", "watch_ratio",
           "avg_watch_seconds", "likes", "comments", "saves", "shares",
           "follows", "profile_visits")

#: Metrics that are rates rather than counts, and are read as percentages when
#: what was typed is bigger than 1. `watch_ratio` is deliberately not one of
#: these: over 1.0 is a real reading there and not a percentage that slipped
#: in, so halving 238% into 2.38 would be reading a loop as a typo.
RATE_METRICS = ("watch_through",)

#: Metrics stored as fractions rather than rounded to whole numbers.
FLOAT_METRICS = ("watch_ratio", "avg_watch_seconds")

#: At and above this, `watch_through` has stopped being a measurement. It is a
#: share of the reel, so it cannot exceed 1.0 - and Instagram's average watch
#: time counts replays, so any reel watched round once lands exactly on the
#: cap. A reel looped twice and a reel watched once through are then the same
#: number, which is why the report declines to rank on it there and says to
#: rank on `watch_ratio` instead.
WATCH_THROUGH_CAP = 1.0

#: Readings kept per reel. A reel is re-read a handful of times over its first
#: day or two and then left alone, so this is a ceiling on a pathological case
#: - somebody running --fetch on a loop - and not a working limit.
MEASUREMENT_HISTORY = 12

#: Why a reel was made and never posted, as a fixed vocabulary.
#:
#: This started as free text, on the reasoning that a sentence you write the
#: same day carries more than a checkbox. It does, to a reader. It carried
#: nothing at all to the report, which never opened the column - and what
#: fourteen of the nineteen binned reels actually said was "not good enough to
#: post", fourteen times. That is a judgement with the evidence left out.
#:
#: A code is comparable across reels, which is the only property that matters
#: here: a rejection is a labelled failure the same day the reel is made, where
#: a view takes a week and a Meta app, so at this sample size the binned column
#: is the fastest signal the project has. `--note` keeps the sentence.
#:
#: Deliberately short. Every code names something a step in this pipeline could
#: be changed to fix, and two of them name no fault in the reel at all.
SKIP_REASONS = {
    "footage": "the footage did not show what was being said",
    "hook": "the opening did not land",
    "claim": "something in it was wrong, or overstated",
    "pacing": "it dragged, or the cuts fought the read",
    "cover": "the cover frame or the text on it was weak",
    "audio": "the read, the music or the mix",
    "duplicate": "too close to a reel already posted",
    "other": "a fault, and none of the above",
    "untested": "never a candidate - a test run, or made before this mattered",
}

#: Codes that are not a verdict on the reel, and are kept out of both tables.
#: A test run was never going to be posted whatever was in it, so counting it
#: as a rejection scores the pipeline's own scaffolding as bad reels; an
#: unlabelled row is a rejection whose reason was lost, which is a row to
#: relabel and not evidence about anything.
UNJUDGED_SKIPS = ("untested", "unlabelled")

#: What a skip written before the vocabulary existed becomes. Not one of
#: SKIP_REASONS: it cannot be chosen, only inherited, and the report prints how
#: many rows are sitting in it and what to type to clear them.
UNLABELLED_SKIP = "unlabelled"

#: The trials PERFORMANCE_EXPLORE can run, and the arms each alternates
#: between. Alternated rather than randomised - see `choose_arm`.
TRIALS = {
    "hook": ("on", "off"),
    "duration": ("short", "long"),
    "structure": ("on", "off"),
    "takeaway": ("on", "off"),
    # The two below are here because `one_sided` named them. Every reel ever
    # made has a bed under it and moving shots in it, so the report asks both
    # questions of the ledger and neither has a second answer to compare with -
    # and the warning that says so pointed at PERFORMANCE_EXPLORE, which until
    # now could not vary either of them. A trial nobody can run is not a
    # remedy.
    "music": ("on", "off"),
    "motion": ("on", "off"),
    # The only trial that acts at step 0, before a word is written, and the
    # only one whose arm changes *which topic gets made* rather than how a
    # chosen topic is treated. It is here because the reach exhibit asked a
    # question no table could answer: on the eight reels posted by
    # 2026-09-11 the board's score did not order reach, and `curiosity` -
    # the one component whose job is "will a stranger stop" - had fired on
    # none of the reels that reached anybody. Rewriting the lexicon off eight
    # rows would be inventing a finding; alternating it is how the ledger
    # settles it instead.
    "curiosity": ("on", "off"),
}

#: What can have been playing under a reel when it was actually published.
#: See `PerformanceLedger.note_audio` for why this is a separate fact from
#: `features["music"]` rather than a reading of it.
AUDIO_MODES = ("bed", "silent", "added")

#: The trials whose arm is spent on the topic board, before there is a topic.
#: They are called out separately because a run has to decide them earlier than
#: the rest - `run_once` arms the board, where every other arm is applied once
#: a candidate exists - and because their segment answers a different question:
#: not "was this reel treated better" but "was a better reel picked".
BOARD_TRIALS = ("curiosity",)


class PerformanceError(RuntimeError):
    """Raised when a run cannot be identified, or a provider cannot be used."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _load(path: Path | str | None) -> dict:
    if not path:
        return {}
    try:
        return visuals._read_json(Path(path), {})
    except (OSError, ValueError):
        return {}


# ---------------------------------------------------------------------------
# The features row - what a reel was
# ---------------------------------------------------------------------------

def duration_band(seconds: float | None) -> str:
    """Ten-second bucket. The exact second is not a segment anyone can read."""
    if not seconds:
        return "unknown"
    low = int(float(seconds) // 10 * 10)
    return f"{low}-{low + 10}s"


def component_names(components) -> list[str]:
    """The names that fired, in the order they were scored.

    Names only. The values are this run's arithmetic under this run's weights,
    and keeping them would invite a later report to average two numbers that
    were never on the same scale.
    """
    if not isinstance(components, dict):
        return []
    return [str(name) for name, value in components.items() if value]


def _chosen_hook_components(hook: dict) -> dict:
    """The components belonging to the opening that was actually used."""
    chosen = hook.get("chosen")
    for candidate in hook.get("candidates") or []:
        if isinstance(candidate, dict) and candidate.get("text") == chosen:
            return candidate.get("components") or {}
    return {}


#: The scored decisions a reel carries the components of, and the column each
#: is stored in. Every one of them is a set of names that fired, asked as a
#: question with two answers by `facets`.
_COMPONENT_KINDS = (
    ("topic", "topic_components"),
    ("hook", "hook_components"),
    ("structure", "structure_components"),
    ("takeaway", "takeaway_components"),
    # Not a scored decision like the four above it - a tag either went on the
    # post or it did not - but the same shape of question, and `build_vocabulary`
    # already drops any name that fired on every reel or on none. A tag used
    # twice is asked about and never read, which is what `readable` is for.
    ("hashtag", "hashtag_names"),
    # Not a scored decision either, and not a set that fired: a list of what
    # step 5b could see wrong with the rendered frames. Same shape of question
    # all the same - "did the reels with unreadable captions reach fewer
    # people" is asked exactly like "did the reels tagged #ecology".
    ("look", "look_problems"),
)


def _written_since(features: dict, version: int) -> bool:
    """Whether this row is new enough to carry `version`'s columns.

    An older row does not answer the questions those columns ask. "Not
    rewritten" and "there was no structure pass" are the same absence on a
    version 1 row, and counting them together scores reels against reels that
    never had the choice.
    """
    return (features.get("features_version") or 0) >= version


def _final_structure_components(structure: dict) -> dict:
    """The shape components of the script that was actually spoken.

    A rejected rewrite scored too, and its components are on disk beside the
    ones that were kept. Recording those instead would label the reel with a
    script nobody ever heard.
    """
    if structure.get("revised"):
        return structure.get("revised_components") or {}
    return structure.get("components") or {}


def _final_takeaway_components(takeaway: dict) -> dict:
    """The take-away components of the script that was actually spoken.

    Same rule as `_final_structure_components`, same reason: a rejected rewrite
    scored too, and recording its components would label the reel with a script
    nobody ever heard.
    """
    if takeaway.get("revised"):
        return takeaway.get("revised_components") or {}
    return takeaway.get("components") or {}


def features_for(record: dict) -> dict:
    """Flatten one finished run into the row a report can read.

    Everything here is already on disk in one manifest or another. What this
    adds is a single denormalised row per reel that does not go stale: the run
    directory can be pruned, the weights can be retuned and step 4 can change
    what it writes, and the history still reads.
    """
    run_dir = Path(record.get("run_dir") or "")
    has_dir = bool(run_dir.name)
    assembly = _load(run_dir / "assembly.json" if has_dir else None)
    post = _load(run_dir / "post.json" if has_dir else None)

    script_path = record.get("script_path") or ""
    script = _load(Path(script_path).with_suffix(".json") if script_path else None)
    hook = script.get("hook") if isinstance(script.get("hook"), dict) else {}
    shape = (script.get("structure")
             if isinstance(script.get("structure"), dict) else {})
    keep = (script.get("takeaway")
            if isinstance(script.get("takeaway"), dict) else {})

    def _section(source: dict, name: str) -> dict:
        value = source.get(name)
        return value if isinstance(value, dict) else {}

    stats = _section(assembly, "stats")
    captions = _section(assembly, "captions")
    music = _section(assembly, "music")
    cover = _section(assembly, "cover")
    seen = _section(assembly, "look")
    cover_seen = cover.get("look") if isinstance(cover.get("look"), dict) else {}
    tiers = _section(post, "hashtag_tiers")

    bands: dict[str, int] = {}
    for band in tiers.values():
        bands[str(band)] = bands.get(str(band), 0) + 1

    spoken = record.get("duration_seconds") or assembly.get("duration_seconds")

    return {
        "features_version": FEATURES_VERSION,
        "run_id": record.get("run_id") or "",
        "day": record.get("day") or date.today().strftime(config.VIDEO_RUN_DATE_FORMAT),
        "created_at": record.get("finished_at") or record.get("started_at") or _now(),
        "run_dir": str(run_dir) if has_dir else "",
        "video_path": record.get("video_path") or "",
        # The only one of these three that survives the run directory being
        # pruned, because output/logs/ does not sit under it. Every reel binned
        # before this column existed lost its evidence along with its
        # directory, which is why rows sit in `unlabelled` with nothing left to
        # read and no way to settle what went wrong. Not a facet - `facets`
        # asks from an allowlist, and a path is not a question - so this adds a
        # column without adding a version.
        "log_path": record.get("log_path") or "",

        "topic": record.get("topic") or "",
        "topic_source": record.get("topic_source") or "",
        "topic_score": record.get("topic_score"),
        "topic_components": component_names(record.get("topic_components")),

        "target_seconds": record.get("target_seconds"),
        "spoken_seconds": round(float(spoken), 2) if spoken else None,
        "duration_band": duration_band(spoken),

        "hook": hook.get("chosen") or hook.get("original") or "",
        "hook_replaced": bool(hook.get("replaced")),
        "hook_components": component_names(_chosen_hook_components(hook)),
        "hook_card": bool(assembly.get("hook_card")),

        "structure_revised": bool(shape.get("revised")),
        "structure_components": component_names(
            _final_structure_components(shape)),

        "takeaway_revised": bool(keep.get("revised")),
        "takeaway_components": component_names(
            _final_takeaway_components(keep)),

        "shots": stats.get("shots"),
        "beats": stats.get("beats"),
        "mean_shot_seconds": stats.get("mean_shot_seconds"),
        "first_shot_seconds": stats.get("first_shot_seconds"),
        "assets_sourced": stats.get("assets_sourced"),
        # How many beats were filled from the video's single neutral shot.
        # None on every row before version 7, which is why the facet below is
        # gated: a row that never counted this is not a row that counted zero.
        "beats_from_neutral": stats.get("beats_from_neutral"),
        "moving_shots": stats.get("moving_shots") or 0,
        "sfx_cues": stats.get("sfx_cues") or 0,
        # "file", not "name": `Track.to_dict` writes the filename under "file"
        # and has no "name" key at all, so this read None on every reel that
        # had a bed playing. See FEATURES_VERSION 3.
        "music": (music.get("file") or "") if music else "",

        "aligner": captions.get("backend") or "",
        "caption_loss": captions.get("loss"),

        # Only when the cover was actually written: a run that failed before
        # step 5 got there has no cover, which is not the same fact as a cover
        # that came from the footage.
        "cover_source": (str(cover.get("source") or "")
                         if cover.get("written") else ""),

        # Step 5b. `look_graded` says whether anything looked at all, and is
        # not redundant with the three below it: every reel made before this
        # existed, and every reel whose look failed or was switched off, has a
        # None grade and no problems - which reads identically to a clean one.
        "look_graded": bool(seen.get("graded")),
        "look_grade": seen.get("grade"),
        "look_problems": sorted({
            str(name).strip().lower() for name in (seen.get("problems") or [])
            if str(name).strip()
        }),
        # The cover is graded separately and against a different question -
        # "is this worth a tap", not "is this worth posting" - so it is its own
        # column. It is also the one number here that moved a decision rather
        # than only describing one, which is what makes it worth correlating:
        # if the covers the model liked reach no better, the weight is wrong.
        "cover_look_grade": (cover_seen.get("grade")
                             if cover.get("written") else None),

        "post_mode": post.get("mode") or "",
        "hashtags": len(post.get("hashtags") or []),
        # Normalised to bare lowercase, because the same tag reaches here as
        # "#WoodWideWeb" and "#woodwideweb" depending on which pass wrote it,
        # and two spellings of one tag are two facets that each halve their
        # own evidence.
        "hashtag_names": sorted({
            str(tag).strip().lstrip("#").lower()
            for tag in (post.get("hashtags") or []) if str(tag).strip()
        }),
        "tag_bands": bands,

        "trial": record.get("trial") or {},
        "warnings": len(record.get("warnings") or []),
    }


def write_features(features: dict, run_dir: Path | str | None = None) -> list[str]:
    """Write features.json beside the reel. A warning, never a failure.

    The ledger is what the report reads; this copy exists because every other
    step leaves its manifest in the run directory, and a run directory that
    cannot explain itself is a worse artifact.
    """
    directory = Path(run_dir or features.get("run_dir") or "")
    if not directory.name:
        return []
    try:
        visuals._write_json(directory / FEATURES_FILENAME, features)
    except OSError as exc:
        return [f"could not write {FEATURES_FILENAME} to {directory}: {exc}"]
    return []


def record_from_dir(run_dir: Path | str) -> dict:
    """A run record shaped from what a finished run directory still holds.

    Step 9 arrives after reels have already been made, and on the day it is
    installed the history sitting in output/ is the entire sample. So a run
    directory is read the way step 8 would have described it at the time.

    It knows less than a live run does - step 0's scoring is not written into
    the run directory, so a backfilled reel has no topic components and no
    source - and the row says so about itself rather than leaving a later
    report to quietly treat "no components" as "no components fired".
    """
    directory = Path(run_dir)
    assembly = _load(directory / "assembly.json")
    audio = assembly.get("audio_path") or ""
    script_path = ""
    if audio:
        stem = Path(audio).stem
        candidate = config.SCRIPTS_DIR / f"{stem}.txt"
        script_path = str(candidate)

    return {
        "run_id": directory.name,
        "run_dir": str(directory),
        "day": directory.parent.name,
        "started_at": assembly.get("created_at") or "",
        "finished_at": assembly.get("created_at") or "",
        "topic": assembly.get("topic") or "",
        "duration_seconds": assembly.get("duration_seconds"),
        "video_path": assembly.get("video_path")
                      or str(directory / config.VIDEO_FILENAME),
        "script_path": script_path,
    }


# ---------------------------------------------------------------------------
# The ledger - what a reel did
# ---------------------------------------------------------------------------

def _parse_stamp(value) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(str(value))
    except ValueError:
        try:
            parsed = datetime.strptime(str(value)[:10], "%Y-%m-%d")
        except ValueError:
            return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def normalise_metrics(raw: dict) -> tuple[dict, list[str]]:
    """Read what was typed. A percentage and a fraction are both accepted.

    `--watched 61` and `--watched 0.61` mean the same thing, because both are
    what a person reads off the insights screen depending on which number it
    happened to show them.
    """
    metrics: dict[str, float] = {}
    warnings: list[str] = []
    for name, value in (raw or {}).items():
        if name not in METRICS or value is None:
            continue
        try:
            number = float(value)
        except (TypeError, ValueError):
            warnings.append(f"{name}: {value!r} is not a number - ignored")
            continue
        if number < 0:
            warnings.append(f"{name}: {number:g} is negative - ignored")
            continue
        if name in RATE_METRICS:
            if number > 1.0:
                number /= 100.0
            if number > 1.0:
                warnings.append(f"{name}: over 100% - clamped")
                number = 1.0
            metrics[name] = round(number, 4)
        elif name == "avg_watch_seconds":
            metrics[name] = round(number, 2)
        elif name in FLOAT_METRICS:
            metrics[name] = round(number, 4)
        else:
            metrics[name] = int(number)

    # Typing one of the two watch numbers is typing both: they are the same
    # division either side of the cap, and a row carrying only `watch_through`
    # cannot be ranked once it sits on it.
    if "watch_ratio" not in metrics and "watch_through" in metrics:
        metrics["watch_ratio"] = metrics["watch_through"]
    return metrics, warnings


def resolve_skip_reason(text: str) -> tuple[str, str, list[str]]:
    """Read what was typed as (code, note, warnings).

    An exact code, an unambiguous prefix of one, or - for anything else - the
    unlabelled bucket with the sentence kept as the note. Free text is not
    rejected: the sentence someone wrote is still the better record of what was
    wrong with a reel, and refusing it would lose that to save a keystroke.
    """
    typed = (text or "").strip()
    if not typed:
        return UNLABELLED_SKIP, "", []
    folded = typed.lower()
    if folded in SKIP_REASONS:
        return folded, "", []

    matches = [code for code in SKIP_REASONS if code.startswith(folded)]
    if len(matches) == 1:
        return matches[0], "", []
    if len(matches) > 1:
        return UNLABELLED_SKIP, typed, [
            f"{typed!r} could be any of {', '.join(sorted(matches))} - "
            f"recorded as {UNLABELLED_SKIP}"
        ]
    return UNLABELLED_SKIP, typed, [
        f"{typed!r} is not one of {', '.join(SKIP_REASONS)} - recorded as "
        f"{UNLABELLED_SKIP} and kept as the note"
    ]


class PerformanceLedger:
    """One entry per reel: what it was, and what it did.

    The features row is copied in here rather than referenced, so the report is
    one file read and survives run directories being pruned. That duplicates
    features.json deliberately - the same trade step 5 already makes when
    assembly.json restates what the visuals manifest said.
    """

    def __init__(self, path: Path | str | None = None):
        self.path = Path(path or config.PERFORMANCE_FILE)
        self._data = visuals._read_json(
            self.path, {"version": LEDGER_VERSION, "reels": {}}
        )
        self._data.setdefault("version", LEDGER_VERSION)
        if not isinstance(self._data.get("reels"), dict):
            self._data["reels"] = {}
        self.migrated = self._migrate()

    def _migrate(self) -> int:
        """Bring rows written by an older version up to date. Idempotent.

        Everything here is derived from what the row already holds, never
        guessed. A skip written before the vocabulary existed keeps its
        sentence as the note and lands in the unlabelled bucket rather than
        being sorted into a code nobody chose; `watch_ratio` is recomputed from
        the average watch time and the reel's own length, both of which have
        been on every posted row since the provider was first switched on.
        """
        touched = 0
        for run_id, entry in self.reels.items():
            if not isinstance(entry, dict):
                continue
            changed = False

            if entry.get("status") == "skipped" and not entry.get("skip_code"):
                entry["skip_code"] = UNLABELLED_SKIP
                entry.setdefault("skip_note", entry.get("skip_reason") or "")
                changed = True

            metrics = entry.get("metrics")
            if isinstance(metrics, dict) and "watch_ratio" not in metrics:
                length = (entry.get("features") or {}).get("spoken_seconds")
                seconds = metrics.get("avg_watch_seconds")
                if length and seconds is not None and float(length) > 0:
                    metrics["watch_ratio"] = round(
                        float(seconds) / float(length), 4
                    )
                    changed = True
                elif metrics.get("watch_through") is not None:
                    # No seconds to divide, so the capped figure is all there
                    # is. Equal below the cap, and honestly unknown above it.
                    metrics["watch_ratio"] = metrics["watch_through"]
                    changed = True

            if changed:
                entry["run_id"] = entry.get("run_id") or run_id
                touched += 1

        self._data["version"] = LEDGER_VERSION
        return touched

    @property
    def reels(self) -> dict:
        return self._data["reels"]

    @property
    def audience(self) -> dict:
        """The account's own size at the last read, or empty.

        One block for the file rather than a column per reel. It is a
        property of the account and not of any reel, and copying it onto
        twenty-six rows would only invite reading it as though each reel
        had its own - which is the mistake the block exists to stop.
        """
        found = self._data.get("account")
        return found if isinstance(found, dict) else {}

    def set_audience(self, audience: dict) -> None:
        """Record what the provider said about the account. Empty is a no-op.

        A provider that could not read the follower count must not erase a
        figure an earlier read did get: not knowing today is not evidence
        that yesterday was wrong.
        """
        if audience:
            self._data["account"] = dict(audience)

    def entry(self, run_id: str) -> dict:
        entry = self.reels.get(run_id)
        return entry if isinstance(entry, dict) else {}

    def register(self, features: dict) -> None:
        """Note that a reel exists. Never touches an outcome already recorded."""
        run_id = features.get("run_id") or ""
        if not run_id:
            return
        entry = self.reels.setdefault(run_id, {"status": "pending"})
        entry["run_id"] = run_id
        entry["features"] = features
        entry.setdefault("status", "pending")
        entry.setdefault("skip_reason", "")
        entry["made_at"] = features.get("created_at") or _now()

    def record(self, run_id: str, metrics: dict, *, posted_at: str = "",
               source: str = "manual") -> list[str]:
        """Record what a posted reel did. Merges, so numbers can arrive twice."""
        clean, warnings = normalise_metrics(metrics)
        if not clean:
            return warnings + ["no usable numbers were given"]

        entry = self.reels.setdefault(run_id, {"run_id": run_id})
        entry["status"] = "posted"
        entry["skip_reason"] = ""
        entry["metrics"] = {**(entry.get("metrics") or {}), **clean}
        entry["source"] = source
        entry["recorded_at"] = _now()

        known = posted_at or entry.get("posted_at") or ""
        assumed = False
        if not known:
            # The only thing actually known here is when the reel was made.
            # Say so rather than inventing a posting time: the age of a
            # measurement is what decides whether two of them are comparable.
            known = (entry.get("features") or {}).get("created_at") or _now()
            assumed = True
        entry["posted_at"] = known
        entry["posted_at_assumed"] = assumed

        posted = _parse_stamp(known)
        measured = _parse_stamp(entry["recorded_at"])
        if posted and measured:
            entry["hours_after_post"] = round(
                (measured - posted).total_seconds() / 3600.0, 1
            )

        # Every reading is kept, not just the last one. The numbers used to be
        # taken once - whatever the reel happened to have the first time
        # anybody looked was what it was judged on forever - and re-reading it
        # would have quietly overwritten the only evidence of how green that
        # first look was. What a reel had at six hours and what it had at two
        # days are two facts, and the pair of them is worth more than either.
        history = entry.get("measurements")
        if not isinstance(history, list):
            history = []
        history.append({
            "recorded_at": entry["recorded_at"],
            "hours_after_post": entry.get("hours_after_post"),
            "source": source,
            "metrics": dict(clean),
        })
        entry["measurements"] = history[-MEASUREMENT_HISTORY:]
        return warnings

    def note_audio(self, run_id: str, mode: str, track: str = "") -> list[str]:
        """Record what was actually playing under the reel when it went out.

        This is the post-time half of the `music` trial, and without it that
        trial measures the opposite of the truth the moment trending audio is
        used. `features["music"]` is the bed step 5 *rendered*, which is the
        only thing the pipeline can know - Instagram's licensed catalogue has
        no publishing API, so a trending track can only be attached by hand in
        the app, after the reel leaves here. A reel rendered with no bed and
        posted with a trending track on it is published *with* music, and the
        `music: off` facet reads it as the silent arm.

        So the arm describes the render and this describes the reel, and
        `report` refuses to let the first stand in for the second.

        Modes:
          bed      the rendered bed is what played - every reel to 2026-09-17
          silent   posted with no music at all
          added    a track was attached in the app; `track` names it

        `track` is kept but never faceted on: one track per reel makes a
        bucket per reel, which is a list of reels wearing a table's clothes.
        """
        mode = (mode or "").strip().lower()
        if mode not in AUDIO_MODES:
            raise PerformanceError(
                f"unknown audio mode {mode!r} - expected one of "
                + ", ".join(sorted(AUDIO_MODES))
            )
        entry = self.reels.setdefault(run_id, {"run_id": run_id})
        entry["audio"] = {"mode": mode, "track": (track or "").strip()}

        warnings: list[str] = []
        if mode == "added" and not (track or "").strip():
            warnings.append(
                "audio recorded as 'added' with no track named - the reel is "
                "counted as having music, but nothing says which"
            )
        if mode != "added" and (track or "").strip():
            warnings.append(
                f"a track was named but the mode is {mode!r}, so it is kept "
                f"and not read as music added in the app"
            )
        return warnings

    #: Everything a row holds only because a provider matched it to a post.
    #: `features` is not here and never can be - it describes the reel itself,
    #: which is true whatever post the reel did or did not become.
    #:
    #: `audio` is not here either, and for a related reason: it is something
    #: the user did to the reel at post time, not something a provider handed
    #: over. A row that turns out to be matched to the wrong post still had a
    #: trending track attached to it or not, and losing the match does not
    #: make that untrue.
    BORROWED = ("media_id", "permalink", "matched_by", "posted_at",
                "posted_at_assumed", "hours_after_post", "metrics",
                "measurements", "source", "recorded_at")

    def disown(self, run_id: str) -> bool:
        """Give back a post this reel turns out not to be, and its numbers.

        A match that loses its post has to lose what came with it. Otherwise
        the row keeps reading as posted, with a reach and a watch ratio taken
        off somebody else's post, and there is nothing in the ledger to say so
        - which is how media 18100751702021813 came to sit under two rows and
        be counted twice in `--report` for four days. Wrong numbers are worse
        than none: no table can tell them from real ones.

        Every reading goes, not just the last, because the whole history was
        read off the wrong post. Numbers typed in by hand are left alone - a
        person saying what a reel did outranks a provider saying it cannot find
        the post - and the row goes back to `pending`, which is a row waiting
        to be matched or labelled, not a verdict about the reel.
        """
        entry = self.reels.get(run_id)
        if not isinstance(entry, dict) or entry.get("source") == "manual":
            return False
        if not any(entry.get(field) for field in self.BORROWED):
            return False
        for field in self.BORROWED:
            entry.pop(field, None)
        if entry.get("status") == "posted":
            entry["status"] = "pending"
            entry["skip_reason"] = ""
        return True

    def skip(self, run_id: str, reason: str = "") -> list[str]:
        """Record that a reel was made and not posted. The free label.

        `reason` is one of SKIP_REASONS. Anything else is kept as the note and
        filed as unlabelled, which is a row to come back to rather than a row
        that has been thrown away.
        """
        code, note, warnings = resolve_skip_reason(reason)
        entry = self.reels.setdefault(run_id, {"run_id": run_id})
        entry["status"] = "skipped"
        entry["skip_code"] = code
        entry["skip_note"] = note
        # Kept, and kept readable: every row before the vocabulary had only
        # this column, and the report and the CLI both still print it.
        entry["skip_reason"] = note or SKIP_REASONS.get(code, code)
        entry["recorded_at"] = _now()
        entry.pop("metrics", None)
        return warnings

    def rows(self) -> list[dict]:
        """Every reel carrying a features row, oldest first."""
        rows = [dict(entry) for entry in self.reels.values()
                if isinstance(entry, dict) and isinstance(entry.get("features"), dict)]
        rows.sort(key=lambda e: str((e.get("features") or {}).get("created_at") or ""))
        return rows

    def pending(self) -> list[dict]:
        return [row for row in self.rows()
                if row.get("status", "pending") == "pending"]

    def unsettled(self, settle_hours: int | None = None) -> list[dict]:
        """Every reel worth asking a provider about.

        The reels with no numbers at all, and the posted ones whose numbers are
        younger than `settle_hours` - a reel read six hours after it went out
        has not finished happening, and asking once was the whole defect here.
        A reel measured past the settle age is left alone: its number is as
        final as this is going to make it, and re-reading it forever would
        spend API calls to watch a figure that no longer moves.
        """
        hours = (config.PERFORMANCE_SETTLE_HOURS
                 if settle_hours is None else settle_hours)
        out = []
        for row in self.rows():
            status = row.get("status", "pending")
            if status == "pending":
                out.append(row)
                continue
            if status != "posted" or hours <= 0:
                continue
            age = row.get("hours_after_post")
            # No age means nothing is known about when it was read, and a
            # reading of unknown age is exactly the one worth taking again.
            if age is None:
                out.append(row)
                continue
            age = float(age)
            # A reading that is already comparable is never asked about again,
            # whatever the settle age says. `report` compares on
            # `hours_after_post` - the age of the *latest* reading - so once a
            # reel has been read inside PERFORMANCE_MIN_AGE_HOURS..WINDOW, the
            # only thing another reading can do is push it back out.
            #
            # This only became reachable when the read was automated. Asked by
            # hand it never came up; asked once a day it is the common case,
            # because consecutive readings are a day apart and the window is
            # 24h wide: a reel read at 29h is read again at 53h, and 53h is
            # outside the window that 29h was inside. The reel would have been
            # comparable for exactly one day and then silently stopped being.
            if age >= config.PERFORMANCE_MIN_AGE_HOURS:
                continue
            if age < hours:
                out.append(row)
        return out

    def trial_count(self, name: str) -> int:
        """How many reels have already run an arm of this trial."""
        return sum(
            1 for entry in self.reels.values()
            if isinstance(entry, dict)
            and ((entry.get("features") or {}).get("trial") or {}).get("name") == name
        )

    def save(self) -> list[str]:
        """Persist. A ledger that cannot be written is a warning, not a failure."""
        try:
            visuals._write_json(self.path, self._data)
        except OSError as exc:
            return [f"could not update the performance ledger at {self.path}: {exc}"]
        return []


# ---------------------------------------------------------------------------
# Finding the reel someone meant
# ---------------------------------------------------------------------------

def known_runs(ledger: PerformanceLedger | None = None) -> dict[str, str]:
    """Every run id this project can see, mapped to its directory.

    Two places are asked, because they are wrong in different directions. The
    ledger knows every reel that was registered; the output tree is ground
    truth and survives the ledger being deleted - which is what installing this
    step looks like on a project that has been making reels for a fortnight.
    """
    found: dict[str, str] = {}
    for pattern in (f"*/*/{FEATURES_FILENAME}", f"*/*/{config.VIDEO_FILENAME}"):
        try:
            for path in sorted(config.OUTPUT_DIR.glob(pattern)):
                found.setdefault(path.parent.name, str(path.parent))
        except OSError:
            continue
    for run_id, entry in (ledger.reels if ledger else {}).items():
        if isinstance(entry, dict):
            found.setdefault(run_id, (entry.get("features") or {}).get("run_dir") or "")
    return found


def resolve_run(reference: str, *, ledger: PerformanceLedger | None = None) -> str:
    """Turn what someone typed into a run id.

    Accepts the whole id, the short hash on the end of it, any unique
    substring, a path to the run directory, or "latest".
    """
    text = (reference or "").strip().strip("\"'")
    if not text:
        raise PerformanceError("no reel was given")

    runs = known_runs(ledger)
    if not runs:
        raise PerformanceError(
            "no reels found yet - make one with python pipeline/run.py"
        )

    if text.lower() in ("latest", "last"):
        return max(runs, key=lambda run_id: (runs[run_id] or "", run_id))

    if text in runs:
        return text
    if Path(text).name in runs:
        return Path(text).name

    matches = sorted(run_id for run_id in runs if text.lower() in run_id.lower())
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise PerformanceError(
            f"no reel matches {text!r} - run python pipeline/perform.py to "
            f"see what is waiting for numbers"
        )
    raise PerformanceError(
        f"{text!r} matches {len(matches)} reels: " + ", ".join(matches[:4])
        + (" ..." if len(matches) > 4 else "")
    )


def features_of(run_id: str, *, ledger: PerformanceLedger | None = None) -> dict:
    """The features row for a run, from the ledger or from its directory."""
    entry = ledger.entry(run_id) if ledger else {}
    if isinstance(entry.get("features"), dict):
        return entry["features"]
    directory = known_runs(ledger).get(run_id) or ""
    return _load(Path(directory) / FEATURES_FILENAME) if directory else {}


def backfill(ledger: PerformanceLedger | None = None) -> tuple[int, list[str]]:
    """Adopt every reel on disk that the ledger has not seen.

    Run on every invocation, because the alternative is a step that reports an
    empty history to a project with a fortnight of reels in output/ - and the
    reels made before this existed are the only ones anybody can record numbers
    against today.

    Reels already in the ledger are left alone, outcome and all.
    """
    ledger = ledger or PerformanceLedger()
    adopted = 0
    for run_id, directory in known_runs(ledger).items():
        if ledger.entry(run_id).get("features") or not directory:
            continue
        path = Path(directory)
        if not path.is_dir():
            continue
        row = _load(path / FEATURES_FILENAME)
        if not row:
            row = features_for(record_from_dir(path))
            row["backfilled"] = True
        ledger.register(row)
        adopted += 1
    return adopted, (ledger.save() if adopted else [])


# ---------------------------------------------------------------------------
# Reading it back
#
# The arithmetic is deliberately the dullest thing in the file: a median, a
# ratio and a count. At one reel a day there is no sample that would support
# anything cleverer, and a regression over twenty rows and twenty components
# would produce confident numbers that mean nothing. What earns its place here
# is the bookkeeping around the arithmetic - detrending, sample floors, and
# printing the n next to every figure so a thin segment cannot be misread.
# ---------------------------------------------------------------------------

def saturated(entry: dict) -> bool:
    """Whether this reel's `watch_through` has hit the cap and stopped saying
    anything.

    Instagram's average watch time counts replays, so a reel watched round once
    divides out above 1.0 and a share of the reel cannot be. Measured on the
    reel that prompted this: 89.5s of average watch time on a 37.59s reel is a
    ratio of 2.38 and a watch-through of exactly 1.00 - the highest score in
    the ledger, awarded to the reel with three views. `watch_ratio` carries the
    2.38.
    """
    metrics = entry.get("metrics")
    if not isinstance(metrics, dict):
        return False
    for name in ("watch_ratio", "watch_through"):
        value = metrics.get(name)
        if value is None:
            continue
        try:
            return float(value) >= WATCH_THROUGH_CAP
        except (TypeError, ValueError):
            return False
    return False


def metric_value(entry: dict, metric: str) -> float | None:
    """One reel's number, or None if it cannot be compared.

    Never measured is one reason. The other is a capped `watch_through`: at the
    cap it is no longer a measurement of anything, so it is refused here rather
    than ranked - one chokepoint, so every caller inherits the refusal.
    """
    metrics = entry.get("metrics")
    if not isinstance(metrics, dict):
        return None
    if metric == "watch_through" and saturated(entry):
        return None
    value = metrics.get(metric)
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


#: The two metrics that are about distribution rather than attention. A viewer
#: who saves a reel is told to expect it again; one who sends it hands the
#: account a viewer it did not have. Both are what the take-away pass was built
#: to move, and neither can be read the way `ratios` reads watch-through.
_KEPT_METRICS = ("saves", "shares")


def was_kept(entry: dict) -> float | None:
    """Whether one reel was saved or sent on at all. 1.0, 0.0, or unmeasured.

    A yes/no and not a count, because at these numbers a count cannot be read.
    `ratios` divides each reel by the median of its neighbours and
    `local_baseline` refuses a median of zero, so a metric that is zero on most
    reels yields no baseline and therefore no reading - forever, and silently.
    Every posted reel in the ledger on 2026-09-09 had 0 saves and 0 shares, so
    setting PERFORMANCE_METRIC=saves would have produced an empty table rather
    than an answer, and an empty table looks the same as "no effect".

    A share - "how often did a reel in this bucket get saved at all" - is
    defined at zero, moves the first time one does, and is the same shape as
    the kept/binned rate that already works.
    """
    metrics = entry.get("metrics")
    if not isinstance(metrics, dict):
        return None
    seen = [metrics.get(name) for name in _KEPT_METRICS]
    if all(value is None for value in seen):
        return None
    total = 0.0
    for value in seen:
        try:
            total += float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            continue
    return 1.0 if total > 0 else 0.0


#: The two metrics that are about the account rather than the reel. A profile
#: visit is a viewer who stopped watching and went looking for who made it; a
#: follow is that same viewer deciding to come back. Everything else in
#: `METRICS` is attention paid to one reel and spent there.
_FOLLOW_METRICS = ("follows", "profile_visits")


def won_a_follow(entry: dict) -> float | None:
    """Whether one reel turned a viewer into a follower. 1.0, 0.0, unmeasured.

    A yes/no for the same reason `was_kept` is one, and the reasoning is worth
    repeating because this column is even sparser: `ratios` divides a reel by
    the median of its neighbours and `local_baseline` refuses a median of
    zero, so a metric that is zero on most reels never produces a baseline and
    therefore never produces a reading - silently, and for as long as the
    account stays small. A rate - "how often did a reel in this bucket win a
    follow at all" - is defined at zero and moves the first time one does.

    `follows` alone would be the truer question, but it is the rarer event by
    some margin, and a profile visit is the same intent one step short of it.
    Either counts, and the report says so in the table's own heading rather
    than leaving the reader to assume it means follows.

    Returns None when neither has ever been read, which is every reel made
    before 2026-09-10: the ledger held the column and no fetch filled it. That
    is unmeasured, not zero, and the difference is the whole reason the table
    can be trusted once it starts printing numbers.
    """
    metrics = entry.get("metrics")
    if not isinstance(metrics, dict):
        return None
    seen = [metrics.get(name) for name in _FOLLOW_METRICS]
    if all(value is None for value in seen):
        return None
    total = 0.0
    for value in seen:
        try:
            total += float(value) if value is not None else 0.0
        except (TypeError, ValueError):
            continue
    return 1.0 if total > 0 else 0.0


#: The share of all reach one reel has to hold before the word "average"
#: stops describing anything here. Not a tuned threshold and not a prior: at
#: more than half, the single best reel outweighs every other reel put
#: together, so the mean necessarily sits above all but one of the numbers it
#: was drawn from. Below that it is a spread; above it, it is one event and
#: some noise.
_NO_MIDDLE_SHARE = 0.5


def reach_exhibit(rows: list[dict], limit: int = 12) -> dict:
    """Every posted reel against the raw reach it got, ranked, with no floors.

    The four tables around this one all ask the same shape of question - does
    answering something one way beat answering it the other - and none of them
    can print until `PERFORMANCE_MIN_SAMPLES` reels have landed on each side.
    Reach does not need them to. It has a reading on every posted reel, it
    spans nought to several hundred, and the one thing worth knowing about it
    - that nearly all of it arrived on one reel - is exactly what the first
    median anyone takes destroys.

    So this is deliberately **not** run through `_segment`, and the three
    filters that guard the tables are all off on purpose:

      * **no freshness window.** The window exists so a ratio compares reels
        of a like age. Nothing here is a ratio. A reel read at 167h did have
        longer to accumulate, which is why `hours` is carried on every row and
        printed beside it rather than used to drop it: at a spread of forty to
        one, read-age is not what separates the top of this list from the
        bottom, and filtering on it would cost most of the observations there
        are to explain a difference it cannot account for.
      * **no reach floor.** `PERFORMANCE_MIN_REACH` keeps a reel out of the
        *ratio* tables because a denominator of three measures whoever opened
        the app. As an observation *of* reach, three is not a weak reading -
        it is the reading, and it is the one the account keeps getting.
      * **no sample floor.** Nothing here is read as an effect, so there is no
        n at which it turns trustworthy. It is the list of what happened, and
        it is labelled an exhibit where it prints so that it is never quoted
        as a finding about any component on it.

    `concentration` is the number this exists to print: the share of all reach
    that the single best reel holds. When one reel carries most of it, every
    mean in this report is describing a distribution with no middle, and "an
    ordinary reel for this account" is not a thing that exists.

    `by_score` re-sorts the same rows by what the board scored them, so the two
    orderings can be read against each other. That comparison is left to the
    reader on purpose - printing a correlation over eight rows, one of which is
    four fifths of the total, would be inventing the finding this is here to
    avoid inventing.

    A reel with no reach reading is counted in `unread` and never listed as a
    zero: never asked is not nobody saw it, the same distinction `won_a_follow`
    makes one stage further down the funnel.
    """
    listed: list[dict] = []
    unread = 0
    for row in rows:
        if row.get("status") != "posted":
            continue
        reach = (row.get("metrics") or {}).get("reach")
        try:
            reach = float(reach) if reach is not None else None
        except (TypeError, ValueError):
            reach = None
        if reach is None:
            unread += 1
            continue
        features = row.get("features") or {}
        components = features.get("topic_components")
        if isinstance(components, dict):
            components = component_names(components)
        listed.append({
            "run_id": row.get("run_id") or "",
            "topic": features.get("topic") or "",
            "source": features.get("topic_source") or "",
            "reach": int(reach),
            "score": features.get("topic_score"),
            "components": [str(name) for name in components]
                          if isinstance(components, list) else [],
            "hours": row.get("hours_after_post"),
        })

    listed.sort(key=lambda entry: -entry["reach"])
    total = sum(entry["reach"] for entry in listed)
    for entry in listed:
        entry["share"] = round(entry["reach"] / total, 3) if total else None

    scored = [entry for entry in listed if entry.get("score") is not None]
    scored.sort(key=lambda entry: -float(entry["score"]))

    return {
        "rows": listed[:limit],
        "by_score": scored[:2],
        "of": len(listed),
        "total": total,
        "unread": unread,
        "concentration": listed[0]["share"] if listed and total else None,
    }


def local_baseline(values: list[float], index: int, span: int) -> float | None:
    """Median of an equal number of reels either side of this one.

    Equal on purpose, and the reason is the whole point of detrending. A window
    that took whatever neighbours it could find would, at the ends of the
    series, be looking only backwards or only forwards - and on an account that
    is growing, a baseline built only from older reels is the difference
    between "this reel did well" and "this reel is recent". The newest reel is
    exactly the one that bias would flatter, and exactly the one you look at.

    So the newest reel has no baseline until the reels after it exist. That is
    the honest answer rather than a flattering one, and it costs a few days.
    """
    reach = min(span // 2, index, len(values) - 1 - index)
    if reach <= 0:
        return None
    nearest = values[index - reach:index] + values[index + 1:index + 1 + reach]
    middle = statistics.median(nearest)
    return middle if middle > 0 else None


def ratios(entries: list[dict], metric: str, span: int) -> dict[str, float]:
    """Each posted reel's number as a ratio to its neighbours' median.

    1.0 is an ordinary reel for the account at that point in its life. 1.3 did
    30% better than the reels around it. Raw numbers never leave this function,
    because an account that is growing makes every later reel look better than
    every earlier one whatever was in it.
    """
    measured = [(entry, metric_value(entry, metric)) for entry in entries]
    measured = [(entry, value) for entry, value in measured if value is not None]
    if not measured:
        return {}
    measured.sort(key=lambda pair: str(
        pair[0].get("posted_at") or (pair[0].get("features") or {}).get("created_at") or ""
    ))
    values = [value for _, value in measured]

    out: dict[str, float] = {}
    for index, (entry, value) in enumerate(measured):
        baseline = local_baseline(values, index, span)
        if baseline:
            out[entry.get("run_id") or str(index)] = round(value / baseline, 3)
    return out


def facets(features: dict, vocabulary: dict[str, list[str]]) -> list[tuple[str, str]]:
    """The (question, answer) pairs one reel is evidence for.

    A component is asked as a question with two answers rather than recorded
    only when it fires, because "explainer shape reels do 1.2x" means nothing
    without the reels where it did not fire to compare against.
    """
    pairs: list[tuple[str, str]] = []

    if features.get("topic_source"):
        pairs.append(("topic source", str(features["topic_source"])))
    if features.get("duration_band") and features["duration_band"] != "unknown":
        pairs.append(("length", str(features["duration_band"])))
    pairs.append(("hook pass", "replaced" if features.get("hook_replaced") else "kept"))
    # Only a row written since the structure pass existed can answer for it. On
    # an older one "not rewritten" and "there was no such pass" are the same
    # absence, and counting them together would score reels against reels that
    # never had the choice.
    if _written_since(features, 2):
        pairs.append(("structure pass",
                      "rewritten" if features.get("structure_revised") else "kept"))
    # Version 4 for the same reason again: every earlier row says "kept"
    # because no take-away pass existed to rewrite anything, and counting those
    # as reels that were offered a hand-over and did not need one would put the
    # whole back catalogue in the control arm of a trial it never ran.
    if _written_since(features, 4):
        pairs.append(("take-away pass",
                      "rewritten" if features.get("takeaway_revised") else "kept"))
    # Version 5, and only where a cover was written: every earlier row stores
    # "" whether the cover came from the hook card or from the footage, so
    # asking them would score the whole back catalogue as one answer.
    if _written_since(features, 5) and features.get("cover_source"):
        pairs.append(("cover", str(features["cover_source"])))
    # Version 6, and only where something actually looked: an ungraded reel is
    # not a badly graded one, and every reel in the history before step 5b is
    # ungraded. `look_graded` is asked of version 6 rows so that switching the
    # look off stays visible as its own answer rather than as silence.
    if _written_since(features, 6):
        pairs.append(("looked at",
                      "yes" if features.get("look_graded") else "no"))
        if features.get("look_graded") and features.get("look_grade") is not None:
            pairs.append(("look grade", str(features["look_grade"])))
        if features.get("cover_look_grade") is not None:
            pairs.append(("cover look grade", str(features["cover_look_grade"])))
    # Version 7: the neutral shot is one shot for the whole video, so a reel
    # that filled several beats from it is showing the same footage over and
    # over. The two reels graded 1/3 by the look pass had filled 4 of 8 and 5
    # of 6 beats that way; the one graded 2 had filled 1 of 5. That is three
    # rows and a guess at a mechanism, not a finding - the facet is here so
    # reach can answer it. Banded, because the count is bounded by
    # VISUAL_MAX_NEUTRAL_BEATS and raw counts would split the arms too thin.
    if _written_since(features, 7) and features.get("beats_from_neutral") is not None:
        filled = int(features["beats_from_neutral"])
        pairs.append(("neutral beats",
                      "0" if filled == 0 else "1" if filled == 1 else "2+"))
    if features.get("aligner"):
        pairs.append(("captions", str(features["aligner"])))
    # Only from version 3: every earlier row records "" for the music whether
    # a bed played or not, so asking them scores the whole back catalogue as
    # music-off and buries whatever the column is worth.
    if _written_since(features, 3):
        pairs.append(("music", "on" if features.get("music") else "off"))
    pairs.append(("sfx", "on" if features.get("sfx_cues") else "off"))
    if _written_since(features, 2):
        pairs.append(("motion", "on" if features.get("moving_shots") else "off"))

    trial = features.get("trial") or {}
    if trial.get("name") and trial.get("arm"):
        pairs.append((f"trial: {trial['name']}", str(trial["arm"])))

    for kind, key in _COMPONENT_KINDS:
        if kind == "structure" and not _written_since(features, 2):
            continue
        if kind == "takeaway" and not _written_since(features, 4):
            continue
        # An older row lists no tags at all, and "this reel did not use
        # #ecology" is a different fact from "nobody wrote the tags down".
        if kind == "hashtag" and not _written_since(features, 5):
            continue
        # An ungraded reel has an empty problem list because nobody looked,
        # not because there was nothing to find. Asking it "was this dark" and
        # recording "no" is the version-gating mistake in its purest form.
        if kind == "look" and not features.get("look_graded"):
            continue
        fired = {str(name) for name in features.get(key) or []}
        for name in vocabulary.get(kind, ()):
            pairs.append((f"{kind}: {name}", "fired" if name in fired else "absent"))

    return pairs


def build_vocabulary(rows: list[dict]) -> dict[str, list[str]]:
    """Component names worth asking about: ones that both fired and did not.

    A component present on every reel in the history has no contrast in it and
    would print a segment that compares a set against itself.
    """
    vocabulary: dict[str, list[str]] = {}
    for kind, key in _COMPONENT_KINDS:
        seen: dict[str, int] = {}
        for row in rows:
            for name in (row.get("features") or {}).get(key) or []:
                seen[str(name)] = seen.get(str(name), 0) + 1
        vocabulary[kind] = sorted(
            name for name, count in seen.items() if 0 < count < len(rows)
        )
    return vocabulary


#: Resamples behind each row's interval. A thousand is where the 5th and 95th
#: percentiles of the resampled deltas stop moving in the third decimal, which
#: is the only precision this prints.
_RESAMPLES = 1000

#: Fixed, and not a tuning knob. The report is cached against the ledger's
#: modification time and its numbers get quoted in commit messages, so the same
#: ledger has to produce the same interval twice. Which integer it is changes
#: nothing about what the interval means.
_RESAMPLE_SEED = 20260911


def _interval(mine: list[float], rest: list[float], statistic,
              confidence: float, seed: int) -> tuple[float | None, float | None]:
    """A confidence interval on the delta `_segment` is about to print.

    **Why a bootstrap and not a t-test.** `statistic` is the median by default,
    and a t-test has nothing to say about a median. The spread it would be
    asked to assume is normal is one where a single reel holds 76% of all the
    reach in the file. Resampling the numbers actually in the bucket makes
    neither assumption and works for whatever `statistic` is - including the
    mean the rate tables pass in - at the cost of a thousand medians per row,
    which is microseconds on buckets this size.

    Returns `(None, None)` when either side has fewer than two readings, which
    is not a weak interval but an absent one: a bucket of one resamples to
    itself every time and would report a width of exactly nought - the most
    confident row in the table, drawn from the least evidence.
    """
    if len(mine) < 2 or len(rest) < 2:
        return (None, None)
    rng = random.Random(seed)
    pick, count_mine, count_rest = rng.choice, len(mine), len(rest)
    deltas = sorted(
        statistic([pick(mine) for _ in range(count_mine)])
        - statistic([pick(rest) for _ in range(count_rest)])
        for _ in range(_RESAMPLES)
    )
    # Clamped, because a confidence outside (0, 1) indexes the resamples from
    # the wrong end: 1.5 makes `tail` negative, and a negative index quietly
    # returns a percentile from the other tail rather than raising.
    tail = (1.0 - min(max(confidence, 0.5), 0.999)) / 2.0
    last = len(deltas) - 1
    return (round(deltas[int(tail * last)], 3),
            round(deltas[int((1.0 - tail) * last)], 3))


def _segment(rows: list[dict], score, vocabulary: dict[str, list[str]],
             min_samples: int, statistic=statistics.median,
             extra=None) -> list[dict]:
    """Group reels by facet and compare each answer against the rest.

    `score` maps a row to a number, or to None where the row cannot be scored.
    `statistic` summarises a bucket: the median for performance, where one
    freak reel should not carry a segment, and the mean for a rate like "how
    often did this get posted", where the median of a column of ones and zeros
    is a majority vote rather than a share.

    `extra` adds questions that are not about the reel but about what was done
    with it - the time of day it went out - and is per table, because those are
    not answerable by every table. See `_when_posted`.
    """
    buckets: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        value = score(row)
        if value is None:
            continue
        pairs = facets(row.get("features") or {}, vocabulary)
        if extra:
            pairs = pairs + list(extra(row))
        for pair in pairs:
            buckets.setdefault(pair, []).append(value)

    questions: dict[str, list[str]] = {}
    for question, answer in buckets:
        questions.setdefault(question, []).append(answer)

    out: list[dict] = []
    # Counted rather than derived from the row, so the same ledger seeds the
    # same resamples in the same order however the table is sorted afterwards.
    index = 0
    for question, answers in questions.items():
        for answer in sorted(set(answers)):
            mine = buckets[(question, answer)]
            rest = [value for other in set(answers) if other != answer
                    for value in buckets[(question, other)]]
            here = statistic(mine)
            there = statistic(rest) if rest else None
            low, high = _interval(mine, rest, statistic,
                                  config.PERFORMANCE_CONFIDENCE,
                                  _RESAMPLE_SEED + index)
            index += 1
            out.append({
                "question": question,
                "answer": answer,
                "n": len(mine),
                "median": round(here, 3),
                "against": round(there, 3) if there is not None else None,
                "delta": round(here - there, 3) if there is not None else None,
                "readable": len(mine) >= min_samples and len(rest) >= min_samples,
                "of": len(mine) + len(rest),
                # The interval on `delta`, and whether it clears nought. These
                # are the two that say what `readable` only implies: `readable`
                # is a count of reels, and a count of reels is not evidence of
                # anything on its own. A row that is readable and not separated
                # is a real observation with a real answer - this many reels
                # cannot see a difference this small - and it is a different
                # answer from "not enough reels yet", which is what the table
                # used to print in its place.
                "low": low,
                "high": high,
                "separated": bool(low is not None and high is not None
                                  and (low > 0.0 or high < 0.0)),
                # How many answers this question has ever had here. One means
                # every reel in this table answered it the same way, and the
                # row is not a segment at all - there is nothing on the other
                # side to compare against, and no number of further reels
                # makes one appear. `readable` is already False for those, but
                # False covers both "not enough yet" and "not ever", and only
                # one of those is fixed by waiting. See `one_sided`.
                "sides": len(set(answers)),
            })
    out.sort(key=lambda entry: (not entry["readable"], not entry["separated"],
                                -abs(entry["delta"] or 0.0), entry["question"]))
    return out


def one_sided(rows: list[dict]) -> list[str]:
    """The questions every reel in a table answered the same way.

    These are the columns the pipeline holds constant. A segment needs two
    answers to compare, so a constant column can never produce a finding: it
    is not a thin row, it is an absent experiment, and it stays absent until
    something upstream varies. Reported by name because which ones they are is
    the actionable part - a table full of them means the reels are identical,
    not that the choices in them do not matter.

    A question only one reel has answered is skipped. One reel answers every
    question exactly one way, so on a table of one the whole vocabulary comes
    back "constant" - which says nothing about the pipeline and would bury the
    handful of columns that really are held fixed under thirty that are not.
    """
    return sorted({row["question"] for row in rows
                   if row.get("sides") == 1 and row.get("of", 0) >= 2})


def skip_tally(rows: list[dict]) -> list[dict]:
    """Why the binned reels were binned, commonest first.

    Deliberately a tally and not a segment. With fourteen rejections spread
    over eight codes, crossing them with the component vocabulary would produce
    a table of segments of one that all look like findings - the exact failure
    `_segment` refuses elsewhere. A count of "six of these were binned for the
    footage" is a small claim that is actually supported, and it points at the
    step to go and change.

    `share` is of the binned reels that carry a verdict, not of every reel: the
    question this answers is what goes wrong, given that something did.
    """
    judged = [row for row in rows if row.get("status") == "skipped"
              and row.get("skip_code") not in UNJUDGED_SKIPS]
    counts: dict[str, int] = {}
    notes: dict[str, list[str]] = {}
    for row in judged:
        code = str(row.get("skip_code") or UNLABELLED_SKIP)
        counts[code] = counts.get(code, 0) + 1
        note = str(row.get("skip_note") or "").strip()
        if note:
            notes.setdefault(code, []).append(note)

    def distinct(seen: list[str]) -> list[str]:
        """The last few notes on a code, without repeating one.

        A batch relabel writes the same sentence onto every row it touches, and
        printing it three times says no more than printing it once while
        crowding out the notes that differ - which are the ones worth reading.
        """
        out: list[str] = []
        for note in reversed(seen):
            if note not in out:
                out.append(note)
            if len(out) >= 3:
                break
        return list(reversed(out))

    total = len(judged)
    out = [{
        "code": code,
        "means": SKIP_REASONS.get(code, code),
        "n": count,
        "share": round(count / total, 3) if total else 0.0,
        "notes": distinct(notes.get(code, [])),
    } for code, count in counts.items()]
    out.sort(key=lambda row: (-row["n"], row["code"]))
    return out


def posting_slot(entry: dict) -> str:
    """Which six-hour block of the day, UTC, a reel went out in - or "".

    UTC and said so in the label, because the alternative is worse. Converting
    to the account's own clock needs a timezone this does not have and would
    have to be told, and a mislabelled hour is a finding about the wrong time
    of day. The blocks are wide enough that the offset shifts a reel by at most
    one of them.

    A row whose posting time was assumed gets no slot at all. `posted_at` falls
    back to when the reel was *made* when nobody said otherwise, and a reel
    made at noon and posted in the evening would otherwise answer this question
    with a time nothing happened at - the one kind of wrong answer that looks
    exactly like a right one.
    """
    if entry.get("posted_at_assumed"):
        return ""
    stamp = _parse_stamp(entry.get("posted_at"))
    if not stamp:
        return ""
    start = (stamp.astimezone(timezone.utc).hour // 6) * 6
    return f"{start:02d}-{start + 6:02d} utc"


def _when_posted(row: dict) -> list[tuple[str, str]]:
    """The posting-time facet, for the tables where it is not a trick question.

    Never asked of KEPT OR BINNED. A binned reel has no posting time, so there
    the question would separate exactly the posted reels from exactly the
    binned ones and report a perfect correlation with being posted - which is
    the tautology the table is made of, dressed as a finding about the clock.
    """
    slot = posting_slot(row)
    return [("posted", slot)] if slot else []


def _audio_posted(row: dict) -> list[tuple[str, str]]:
    """What was playing under the reel as published, for the tables that can ask.

    Never asked of KEPT OR BINNED, for `_when_posted`'s reason: a binned reel
    was never published, so the question would separate the posted rows from
    the binned ones and report the tautology the table is made of.

    A row that has not been told stays out entirely rather than defaulting to
    "bed". Every reel to 2026-09-17 did play its rendered bed, so defaulting
    would be right about all of them and silently wrong about the first one
    that is not - which is the failure this whole field exists to prevent.
    """
    audio = row.get("audio")
    if not isinstance(audio, dict):
        return []
    mode = str(audio.get("mode") or "").strip().lower()
    return [("audio as posted", mode)] if mode in AUDIO_MODES else []


def audio_disagrees(row: dict) -> str:
    """Why this row's trial arm no longer describes the reel, or "".

    The `music` trial alternates whether step 5 lays a bed. That is a fact
    about the render, and it stops being a fact about the reel the moment a
    track is attached in the app. A row in the `off` arm with `added` audio was
    published with music on it, and counting it as the silent arm does not
    weaken the trial - it inverts it.
    """
    audio = row.get("audio")
    if not isinstance(audio, dict):
        return ""
    mode = str(audio.get("mode") or "").strip().lower()
    trial = ((row.get("features") or {}).get("trial") or {})
    if trial.get("name") != "music":
        return ""
    arm = str(trial.get("arm") or "").strip().lower()
    if arm == "off" and mode == "added":
        return ("was rendered with no bed and posted with a track added in "
                "the app, so it is music-on however the arm is labelled")
    if arm == "on" and mode == "silent":
        return ("was rendered with a bed and recorded as posted silent, which "
                "the arm cannot be describing")
    return ""


def _post_time_facets(row: dict) -> list[tuple[str, str]]:
    """Everything asked of a row about what was *done* with the reel.

    `_segment` takes one `extra`, and there are two of these now - when it went
    out, and what was under it. They are joined here rather than by widening
    `extra`, because they share the one rule that matters: both are questions
    only a published reel can answer, so both are per-table and neither is ever
    asked of KEPT OR BINNED.
    """
    return _when_posted(row) + _audio_posted(row)


def unlabelled_bins(rows: list[dict]) -> list[dict]:
    """The binned reels whose reason was lost, oldest first.

    `UNJUDGED_SKIPS` holds two codes and only one of them is a chore. A test
    run is finished business - it was never a candidate, and no label will ever
    be wanted for it - while an unlabelled row is a verdict somebody reached
    and the ledger failed to keep, which is the only kind of row `--relabel`
    can do anything about.
    """
    return [row for row in rows if row.get("status") == "skipped"
            and str(row.get("skip_code") or UNLABELLED_SKIP) == UNLABELLED_SKIP]


def recover_log(run_id: str, features: dict, logs_dir: Path | str | None = None):
    """The transcript of a run whose row predates the `log_path` column.

    Found by reading output/logs/ for the id rather than by matching on the
    day, because several runs can share a day and only one of them is this
    reel. The files are a few KB each and there are tens of them, so the scan
    is cheaper than the guess is wrong.

    This exists for the rows already in the ledger. Rows written from here on
    carry `log_path` and never reach the scan.
    """
    recorded = str(features.get("log_path") or "")
    if recorded and Path(recorded).is_file():
        return Path(recorded)

    directory = Path(logs_dir or config.LOGS_DIR)
    if not directory.is_dir():
        return None
    # The whole id, then the short form the CLI takes - a log names the run
    # directory in full, but an older one may only carry the suffix people type.
    needles = [run_id]
    tail = run_id.rsplit("-", 1)[-1]
    if tail and tail != run_id:
        needles.append(tail)
    for path in sorted(directory.glob("*.log"), reverse=True):
        try:
            body = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(needle in body for needle in needles):
            return path
    return None


#: Metrics an operator's own accounts inflate simply by opening a reel. A
#: comment is not one of them - own accounts do not comment on every post,
#: and subtracting a flat count from comments would invent negative ones.
#: Comments on a reel that reached nobody are handled by `real_engagement`
#: instead, which discards the whole row rather than shaving a number off it.
SELF_INFLATED = ("reach", "views", "likes")


def net_of_own(metrics: dict, own: int | None = None) -> dict:
    """A reel's numbers with the operator's own accounts taken out.

    Floored at zero rather than allowed negative: `own` is a declared count
    and a reel read before every alt had opened it will sit under it, which
    is a stale reading and not evidence of negative reach.

    Returns a copy. The stored row keeps what Instagram said, so lowering
    PERFORMANCE_OWN_ACCOUNTS later re-reads the same history rather than
    having overwritten it.
    """
    own = config.PERFORMANCE_OWN_ACCOUNTS if own is None else own
    out = dict(metrics or {})
    if own <= 0:
        return out
    for name in SELF_INFLATED:
        value = out.get(name)
        if isinstance(value, (int, float)):
            out[name] = max(0, int(value) - own)
    return out


def real_engagement(rows: list[dict], own: int | None = None) -> dict:
    """What people who are not the operator did, across every posted reel.

    The one summary the ratio tables cannot produce: they compare reels
    against each other, so a column that is the same constant on every row
    reads as no finding at all. Twenty-five likes across eight reels looks
    like a quiet account. Three of them on every reel and one real one is a
    different fact, and it is the fact.

    A reel whose reach does not clear `own` reached **nobody**, so nothing
    it carries can have come from an audience. The whole row is discarded
    rather than netted off, which is why a comment on a reel with a reach of
    three does not survive into the total.
    """
    own = config.PERFORMANCE_OWN_ACCOUNTS if own is None else own
    out = {"reach": 0, "likes": 0, "comments": 0, "saves": 0,
           "shares": 0, "follows": 0, "profile_visits": 0,
           "reels": 0, "unseen": 0, "follow_reels": 0}
    for row in rows:
        if row.get("status") != "posted":
            continue
        metrics = row.get("metrics") or {}
        if not metrics:
            continue
        out["reels"] += 1
        net = net_of_own(metrics, own)
        reach = int(net.get("reach") or 0)
        if reach <= 0:
            out["unseen"] += 1
            continue
        out["reach"] += reach
        out["likes"] += int(net.get("likes") or 0)
        for name in ("comments", "saves", "shares"):
            out[name] += int(metrics.get(name) or 0)
        # Not netted against the operator's own accounts, unlike likes. Those
        # three alts already follow and cannot follow again, so there is no
        # per-reel floor to take out here - a follow counted against a reel is
        # a follow that reel won. Counted separately from `reels` because a
        # reel nothing ever asked for these on is not a reel that won none.
        if any(metrics.get(name) is not None for name in _FOLLOW_METRICS):
            out["follow_reels"] += 1
            for name in _FOLLOW_METRICS:
                out[name] += int(metrics.get(name) or 0)
    return out


def report(*, ledger: PerformanceLedger | None = None, metric: str = "",
           span: int | None = None, min_samples: int | None = None,
           window_hours: int | None = None,
           min_age_hours: int | None = None,
           min_reach: int | None = None) -> dict:
    """Everything the ledger can currently say, and what it cannot say yet.

    Three tables, because three different columns are being read. `kept` uses
    every reel ever made and asks whether you posted it - the free label,
    available from day one. `binned` asks the ones you did not post *why*, and
    is the same free label read a second way. `performance` uses only reels
    with numbers against them.
    """
    ledger = ledger or PerformanceLedger()
    metric = metric or config.PERFORMANCE_METRIC
    span = config.PERFORMANCE_BASELINE_RUNS if span is None else span
    floor = config.PERFORMANCE_MIN_SAMPLES if min_samples is None else min_samples
    window = config.PERFORMANCE_WINDOW_HOURS if window_hours is None else window_hours
    green = (config.PERFORMANCE_MIN_AGE_HOURS
             if min_age_hours is None else min_age_hours)
    reach_floor = (config.PERFORMANCE_MIN_REACH
                   if min_reach is None else min_reach)

    rows = ledger.rows()
    warnings: list[str] = []
    posted = [row for row in rows if row.get("status") == "posted"]
    # A reel that was never a candidate is not a rejection, and an unlabelled
    # one is a rejection with the reason lost. Neither is evidence about a
    # reel, so neither votes in the kept/binned split.
    binned = [row for row in rows if row.get("status") == "skipped"
              and row.get("skip_code") not in UNJUDGED_SKIPS]
    unjudged = [row for row in rows if row.get("status") == "skipped"
                and row.get("skip_code") in UNJUDGED_SKIPS]
    decided = [row for row in rows
               if row.get("status") == "posted"
               or (row.get("status") == "skipped"
                   and row.get("skip_code") not in UNJUDGED_SKIPS)]

    fresh = posted
    if window > 0:
        fresh = [row for row in fresh
                 if row.get("hours_after_post") is None
                 or float(row["hours_after_post"]) <= window]
        if len(fresh) < len(posted):
            warnings.append(
                f"{len(posted) - len(fresh)} reel(s) were measured more than "
                f"{window}h after posting and are not compared with the rest - "
                f"a number read at a week old is not the same measurement"
            )
    if green > 0:
        settled = [row for row in fresh
                   if row.get("hours_after_post") is None
                   or float(row["hours_after_post"]) >= green]
        if len(settled) < len(fresh):
            warnings.append(
                f"{len(fresh) - len(settled)} reel(s) were measured less than "
                f"{green}h after posting and are not compared - a reel read "
                f"the same evening has not finished happening. "
                f"python pipeline/perform.py --fetch reads them again"
            )
        fresh = settled

    # How many people saw it, not how many reels are in the table. A ratio over
    # a reach of three is not a thin measurement of the reel, it is a
    # measurement of who happened to open the app, and averaging those with a
    # reel that reached three hundred puts the two on the same footing. Kept
    # out of the ratio tables for the same reason a reading taken a week late
    # is. Rows with no reach recorded stay in: not knowing the reach is not the
    # same as knowing it was small, and the freshness filters above make the
    # same call about an unknown age.
    # Kept before the reach floor is applied. A reel that reached nobody is a
    # perfectly good observation *of reach* - it is only as a denominator that
    # it is worthless - and filtering the reach table by reach would drop the
    # end of the distribution the account is actually stuck at.
    seen_by = fresh
    unseen: list[dict] = []
    if reach_floor > 0:
        seen = []
        for row in fresh:
            reach = (row.get("metrics") or {}).get("reach")
            (seen if reach is None or float(reach) >= reach_floor
             else unseen).append(row)
        if unseen:
            warnings.append(
                f"{len(unseen)} reel(s) reached fewer than {reach_floor} "
                f"accounts and are not compared - at that reach the ratio "
                f"moves by a third when one person watches, so it measures the "
                f"audience and not the reel. PERFORMANCE_MIN_REACH=0 compares "
                f"them anyway"
            )
        fresh = seen

    vocabulary = build_vocabulary(rows)

    kept = _segment(
        decided,
        lambda row: 1.0 if row.get("status") == "posted" else 0.0,
        vocabulary, floor, statistic=statistics.fmean,
    )
    reasons = skip_tally(rows)

    scores = ratios(fresh, metric, span)
    performance = _segment(
        fresh,
        lambda row: scores.get(row.get("run_id") or ""),
        vocabulary, floor, extra=_post_time_facets,
    )

    # How many people it got in front of, which is upstream of everything else
    # in this report and was the one thing it did not print. Every other table
    # here divides by an audience; this is the audience. Read on 2026-09-09 the
    # seven posted reels reached 289, 64, 9, 6, 3, 3 and 0 accounts - a median
    # of six - and a report whose headline was watch-through was answering a
    # question the account does not have yet.
    reach_scores = ratios(seen_by, "reach", span)
    reached = _segment(
        seen_by,
        lambda row: reach_scores.get(row.get("run_id") or ""),
        vocabulary, floor, extra=_post_time_facets,
    )

    # Read on the same window as the performance table, so every reel is
    # judged at a comparable age: a reel read at 96h has had three more days to
    # be saved than one read at 29h, and a rate mixing the two measures how
    # long ago a reel went out as much as anything about the reel.
    saved = _segment(
        fresh, was_kept, vocabulary, floor, statistic=statistics.fmean,
        extra=_post_time_facets,
    )

    # The last stage of the funnel, and the only one that compounds. Read on
    # the same window as the two above it for the same reason: a reel read at
    # 96h has had three more days to win a follow than one read at 29h.
    followed = _segment(
        fresh, won_a_follow, vocabulary, floor, statistic=statistics.fmean,
        extra=_post_time_facets,
    )

    # The token that fetched all of this, and how long it has left. A
    # provider-agnostic read of whatever the provider chose to say: the key is
    # absent for a hand-typed ledger and for any provider that does not use a
    # token, and absent is silent.
    #
    # It is a warning rather than a line in the table because it is the one
    # thing here that is about the *reading* and not about any reel, and
    # because losing it is total - a lapsed token cannot be refreshed, only
    # replaced from the Meta dashboard, and until it is every daily read does
    # nothing at all.
    token = (ledger.audience or {}).get("token") or {}
    if token.get("warn"):
        left = token.get("days_left")
        warnings.append(
            f"the Instagram token expires in about {left} day(s) and cannot "
            f"be refreshed once it has - python pipeline/instagram.py "
            f"--refresh, then paste the new token into .env as "
            f"INSTAGRAM_ACCESS_TOKEN"
        )

    # What was under the reel as published, and whether the trial still
    # describes it. Both are warnings rather than filters: a row that has not
    # been told simply does not answer the audio facet, and a row whose arm
    # disagrees still carries perfectly good numbers - what it must not do is
    # carry them into a segment labelled with the opposite treatment.
    mislabelled = [(row, audio_disagrees(row)) for row in posted]
    for row, why in mislabelled:
        if why:
            warnings.append(
                f"{row.get('run_id') or '?'} {why} - its `trial: music` "
                f"segment is reading it as the arm it was rendered in, not "
                f"the reel that went out"
            )
    untold = [row for row in posted if not _audio_posted(row)]
    if untold and len(untold) < len(posted):
        warnings.append(
            f"{len(untold)} posted reel(s) have no audio recorded and sit out "
            f"the AUDIO AS POSTED rows - python pipeline/perform.py <run> "
            f"--audio bed|silent|added"
        )

    if not posted:
        warnings.append(
            "no reel has numbers against it yet - "
            "python pipeline/perform.py <run> --views N --watched N"
        )
    elif len(scores) < len(fresh):
        warnings.append(
            f"{len(fresh) - len(scores)} reel(s) have nothing on one side of "
            f"them yet and are not compared - a baseline built only from older "
            f"reels flatters whatever is newest"
        )
    if posted and len(scores) < floor:
        warnings.append(
            f"{len(scores)} reel(s) can be compared, and nothing is read below "
            f"{floor}. This table is a preview of its own shape, not a finding."
        )
    if metric == "watch_through":
        capped = [row for row in fresh if saturated(row)]
        if capped:
            warnings.append(
                f"{len(capped)} reel(s) sit on the watch-through cap and are "
                f"not compared - Instagram's average watch time counts "
                f"replays, so a looped reel and a reel watched once through "
                f"are the same 1.00 here. PERFORMANCE_METRIC=watch_ratio is "
                f"the same number uncapped"
            )
    # A table of zeros and a table with no reels in it print the same way, and
    # they mean opposite things. Said out loud so a flat SAVED OR SENT is never
    # read as "the pass made no difference".
    ever_kept = [row for row in fresh if (was_kept(row) or 0.0) > 0]
    if fresh and not ever_kept:
        warnings.append(
            f"no reel has been saved or sent on yet, so SAVED OR SENT is a "
            f"column of zeros - that is the account's state, not a finding "
            f"about anything in the table. It starts saying something the "
            f"first time one is"
        )

    # Three states, not two, and they mean different things. Never read at all
    # is the one worth shouting about, because it looks exactly like nobody
    # following and is instead nobody asking.
    read_follow = [row for row in fresh if won_a_follow(row) is not None]
    if fresh and not read_follow:
        warnings.append(
            f"no reel has a follows or profile_visits reading on it, so "
            f"FOLLOWED AFTER is empty - that is a metric nothing fetched, not "
            f"an account nobody followed. python pipeline/instagram.py --fetch "
            f"asks for both now; if it warns that this API version does not "
            f"report them, they are account-level only on your token"
        )
    elif read_follow and not [row for row in read_follow
                              if (won_a_follow(row) or 0.0) > 0]:
        warnings.append(
            f"{len(read_follow)} reel(s) have been read for follows and none "
            f"won one, so FOLLOWED AFTER is a column of zeros - that is the "
            f"account's state and not a finding about the table"
        )

    stale = [row for row in rows if row.get("skip_code") == UNLABELLED_SKIP]
    if stale:
        warnings.append(
            f"{len(stale)} binned reel(s) have no reason on them and are read "
            f"by neither table. One word each fixes it: "
            f"python pipeline/perform.py {stale[0].get('run_id', '<run>')} "
            f"--skipped {'|'.join(list(SKIP_REASONS)[:4])}|..."
        )
    if any((row.get("posted_at_assumed") for row in posted)):
        warnings.append(
            "some reels have no posting time recorded, so their age is taken "
            "from when they were made - pass --posted YYYY-MM-DD to correct one"
        )

    # Said with the other warnings because it is the one that no amount of
    # patience clears. Every table above prints "nothing has reached n reels
    # on both sides yet", which invites waiting; for a question the pipeline
    # answers identically every time, waiting is the one thing that cannot
    # work. Read off `kept`, which uses every reel that got a verdict and so
    # sees the widest evidence of what is actually being held fixed.
    flat = one_sided(kept)
    if flat and decided:
        warnings.append(
            f"{len(flat)} question(s) have had one answer on every reel ever "
            f"made ({', '.join(flat[:4])}"
            f"{', ...' if len(flat) > 4 else ''}) - those columns are constant, "
            f"so they cannot produce a finding at any number of reels. "
            f"PERFORMANCE_EXPLORE=" + "|".join(TRIALS) + " alternates one of "
            f"them per reel so there is something to compare"
        )

    return {
        "metric": metric,
        "one_sided": flat,
        "audience": ledger.audience,
        "own_accounts": config.PERFORMANCE_OWN_ACCOUNTS,
        "real": real_engagement(rows),
        "min_reach": reach_floor,
        "unseen": len(unseen),
        "reached": reached,
        "reach_exhibit": reach_exhibit(rows),
        "reach_median": (round(statistics.median(
            [float((row.get("metrics") or {}).get("reach") or 0.0)
             for row in seen_by]), 1) if seen_by else None),
        "reels": len(rows),
        "posted": len(posted),
        "skipped": sum(1 for row in rows if row.get("status") == "skipped"),
        "binned": len(binned),
        "unjudged": len(unjudged),
        "pending": sum(1 for row in rows
                       if row.get("status", "pending") == "pending"),
        "compared": len(scores),
        "min_samples": floor,
        "baseline_runs": span,
        "min_age_hours": green,
        "window_hours": window,
        "kept": kept,
        "reasons": reasons,
        "performance": performance,
        "saved": saved,
        "followed": followed,
        "ever_saved": len(ever_kept),
        "ratios": scores,
        "warnings": warnings,
    }


# ---------------------------------------------------------------------------
# Feeding it back into step 0
# ---------------------------------------------------------------------------

_CACHE: dict = {"key": None, "effects": {}}


def measured_effects(*, ledger: PerformanceLedger | None = None) -> dict[str, dict]:
    """Per topic component, how its reels did against the ones without it.

    {name: {"delta": how much better, "n": on how many reels}} - the n travels
    with the number because a board that printed the effect without it would be
    the one dishonest line on an otherwise self-auditing page.

    **The n is not what says whether to believe it, and this used to think it
    was.** Only rows that are `readable` reach this dict, and `readable` counts
    reels: eight on each side. Drawing two arms of eight from the nine reach
    figures actually in the ledger - 289, 64, 20, 14, 9, 6, 3, 3, 0 - moves the
    difference of medians past +/-5 in 55% of draws with no effect present, so
    an eight-a-side delta arriving here was as likely to be the shape of the
    spread as anything about the component. That delta is not printed and
    forgotten: `adjustment_for` adds it to the score of every future candidate
    carrying the component, which is the ledger teaching step 0 its own noise.
    A row now has to be `separated` as well - its bootstrap interval clear of
    nought - before it is allowed to weigh on a board.

    Cached against the ledger's modification time: this is read once per
    candidate on every board, and re-medianing the history a dozen times to
    print one board would be silly.
    """
    path = Path(config.PERFORMANCE_FILE)
    try:
        key = (str(path), path.stat().st_mtime_ns, config.PERFORMANCE_MIN_SAMPLES)
    except OSError:
        key = (str(path), 0, config.PERFORMANCE_MIN_SAMPLES)
    if ledger is None and _CACHE["key"] == key:
        return _CACHE["effects"]

    data = report(ledger=ledger)
    effects = {
        entry["question"].split(": ", 1)[1]: {"delta": entry["delta"],
                                              "n": entry["n"]}
        for entry in data["performance"]
        if entry["readable"] and entry["separated"] and entry["answer"] == "fired"
        and entry["question"].startswith("topic: ") and entry["delta"] is not None
    }
    if ledger is None:
        _CACHE.update(key=key, effects=effects)
    return effects


def adjustment_for(components, *, ledger: PerformanceLedger | None = None) -> dict:
    """What the ledger would add to a candidate carrying these components.

    Returns {"value", "parts", "note"} - and {"value": 0.0} whenever the answer
    would be made up. A component whose reels ran 30% over their neighbours is
    worth +0.3 here, which is deliberately small beside step 0's hand-set
    weights of 1 to 3: measured evidence at this sample size is a nudge to a
    ranking, not a replacement for it. PERFORMANCE_MAX_ADJUST is the ceiling.
    """
    blank = {"value": 0.0, "parts": {}, "note": ""}
    if not config.PERFORMANCE_FEEDBACK:
        return blank

    ledger = ledger or PerformanceLedger()
    rows = ledger.rows()
    if len(rows) < config.PERFORMANCE_MIN_REELS:
        return blank

    effects = measured_effects(ledger=ledger)
    names = component_names(components) if isinstance(components, dict) else [
        str(name) for name in (components or [])
    ]
    found = {name: effects[name] for name in names if name in effects}
    if not found:
        return blank

    parts = {name: effect["delta"] for name, effect in found.items()}
    ceiling = abs(config.PERFORMANCE_MAX_ADJUST)
    value = round(max(-ceiling, min(ceiling, sum(parts.values()))), 3)
    note = ", ".join(
        f"{name} {1 + effect['delta']:.2f}x, n={effect['n']}"
        for name, effect in sorted(found.items(),
                                   key=lambda kv: -abs(kv[1]["delta"]))
    )
    return {"value": value, "parts": parts, "note": note}


# ---------------------------------------------------------------------------
# Exploration
# ---------------------------------------------------------------------------

def choose_arm(trial: str = "", *, ledger: PerformanceLedger | None = None) -> dict:
    """Which arm of the running trial this reel takes.

    Alternated, not randomised. The pipeline makes one reel a day, so a trial
    is decided on tens of observations, and at that size randomising can hand
    you 19 of one arm and 11 of the other by luck alone - which is not a
    comparison, and there is no second month to spend finding that out.

    A trial exists because without one there is nothing to compare. Left to
    itself the pipeline replaces the hook every single time, and after a year
    the ledger holds 365 reels that all made the same choice and cannot say
    whether it was the right one.
    """
    name = (trial or config.PERFORMANCE_EXPLORE or "off").strip().lower()
    if name in ("", "off", "none"):
        return {}
    arms = TRIALS.get(name)
    if not arms:
        return {}
    index = (ledger or PerformanceLedger()).trial_count(name)
    return {"name": name, "arm": arms[index % len(arms)], "index": index}


def apply_arm(trial: dict, *,
              duration: int | None) -> tuple[int | None, object, object, object]:
    """What a trial arm changes about the reel that is about to be made.

    Returns the duration to ask step 2 for, and what to pass as its `hooks`,
    `structure` and `takeaway` arguments (None leaves config alone, False turns
    that pass off).
    """
    name, arm = (trial or {}).get("name"), (trial or {}).get("arm")
    if name == "hook":
        return duration, (False if arm == "off" else None), None, None
    if name == "structure":
        return duration, None, (False if arm == "off" else None), None
    if name == "takeaway":
        return duration, None, None, (False if arm == "off" else None)
    if name == "duration" and duration:
        step = max(1, config.SCRIPT_DURATION_STEP_SECONDS)
        shifted = duration - step if arm == "short" else duration + step
        return (max(config.SCRIPT_MIN_SECONDS,
                    min(config.SCRIPT_MAX_SECONDS, shifted)), None, None, None)
    return duration, None, None, None


def arm_assembly(trial: dict) -> dict:
    """What a trial arm changes about step 5, as keyword arguments for it.

    Separate from `apply_arm` because the two are spent at different points in
    the run: `apply_arm` answers step 2, before a word is written, and this
    answers step 5, after the footage is on disk. Folding them together would
    mean deciding both an hour apart from one call, and step 5 already takes
    every one of these as an override.

    Anything not named here is left to config, which is what a trial on some
    other question wants.
    """
    name, arm = (trial or {}).get("name"), (trial or {}).get("arm")
    if name == "music":
        # False, not "": step 5 reads False as "no bed" and None as "rotate
        # over assets/music/ as usual".
        return {"music": False} if arm == "off" else {}
    if name == "motion":
        return {"motion": "off"} if arm == "off" else {}
    return {}


def arm_board(trial: dict) -> dict:
    """What a trial arm changes about step 0, as keyword arguments for it.

    The third of these, and the earliest: `arm_board` answers the topic board
    before there is a topic, `apply_arm` answers step 2 once there is one, and
    `arm_assembly` answers step 5 once the footage is down. They are three
    functions and not one because a run spends them hours apart, and because
    only this one can change what the reel is *about* - every other arm takes
    the topic as given and varies the treatment.

    That difference is worth stating plainly, because it changes how the
    resulting segment reads. A `hook` arm compares two ways of opening the
    same kind of reel. A `curiosity` arm compares two different *boards*, and
    what it measures is not "does this wording do better" but "does charging
    this component pick better topics" - which is the question the board's
    weights have never been asked.

    Anything not named here is left to config, which is what a trial on some
    other question wants.
    """
    name, arm = (trial or {}).get("name"), (trial or {}).get("arm")
    if name == "curiosity":
        # False, not 0.0: `score_candidate` reads False as "do not charge it"
        # and None as "charge it at CURIOSITY_SIGNAL_WEIGHT as usual". A zero
        # weight would be a third state nothing else in the file has.
        return {"curiosity": False} if arm == "off" else {}
    return {}


# ---------------------------------------------------------------------------
# Where the numbers come from
# ---------------------------------------------------------------------------

def load_provider(path: str = ""):
    """Import the dotted path in PERFORMANCE_PROVIDER, or return None.

    The callable is handed a list of run ids and the features row for each, and
    should return {run_id: {metric: value}} for whichever ones it could measure.
    Anything it leaves out stays pending, exactly as if it had not been asked.

    There is no provider in this repo, and the default is to type the numbers
    in. The reason is not squeamishness about the API: Instagram insights need
    a Professional account, a Meta app and an OAuth token that expires every
    sixty days, and none of that is a reason to leave the ledger unbuilt for a
    fortnight. It is also the wrong first thing to automate - the column that
    matters most in the first month is `--skipped`, which no API will ever have.
    """
    target = (path or config.PERFORMANCE_PROVIDER or "").strip()
    if not target:
        return None
    module_name, _, attr = target.rpartition(".")
    if not module_name or not attr:
        raise PerformanceError(
            f"PERFORMANCE_PROVIDER must be a dotted path like "
            f"mypackage.module.fetch - got {target!r}"
        )
    try:
        provider = getattr(import_module(module_name), attr)
    except (ImportError, AttributeError) as exc:
        raise PerformanceError(
            f"could not import PERFORMANCE_PROVIDER {target!r}: {exc}"
        ) from exc
    if not callable(provider):
        raise PerformanceError(f"PERFORMANCE_PROVIDER {target!r} is not callable")
    return provider


#: What a provider may write into the ledger about a reel on its own account:
#: which post it became, where it lives, how that was decided, and when it went
#: up. Not metrics - those come back from the call - and not status, which is
#: step 9's to set.
PROVIDER_ANNOTATIONS = ("media_id", "permalink", "matched_by", "posted_at")


def fetch(*, ledger: PerformanceLedger | None = None, provider=None,
          settle_hours: int | None = None) -> tuple[int, list[str]]:
    """Ask the configured provider about every reel whose numbers are not final.

    Which is more than the reels with no numbers at all. A reel read a few
    hours after it went out has a number, and it is the number of a reel that
    is still being shown; asking once and never again is what left the newest
    reel in this ledger sitting on three views. So a posted reel is asked about
    again until its reading is PERFORMANCE_SETTLE_HOURS old, and each answer is
    appended to its history rather than replacing what came before.
    """
    ledger = ledger or PerformanceLedger()
    provider = provider or load_provider()
    if provider is None:
        raise PerformanceError(
            "no PERFORMANCE_PROVIDER is configured - record the numbers by "
            "hand: python pipeline/perform.py <run> --views N --watched N"
        )

    waiting = ledger.unsettled(settle_hours)
    if not waiting:
        return 0, []

    warnings: list[str] = []
    try:
        answers = provider([row["run_id"] for row in waiting],
                           {row["run_id"]: row.get("features") or {}
                            for row in waiting})
    except Exception as exc:  # a provider is third-party code
        raise PerformanceError(f"PERFORMANCE_PROVIDER failed: {exc}") from exc

    # A provider is allowed to write to the ledger for itself - the Instagram
    # one records which post it matched a reel to, how, and when that post went
    # up. It writes that to disk while this object is holding a copy read
    # before it ran, so the fields it owns are merged back in; without this the
    # save below quietly reverts everything the provider recorded.
    written = _load(ledger.path)
    stored = written.get("reels") or {}
    # The account block travels the same way and for the same reason, but
    # it is not in PROVIDER_ANNOTATIONS because it is not a field on a reel
    # - it is one row about the account, and merging it per run_id would
    # write the same follower count onto every row that happened to answer.
    ledger.set_audience(written.get("account") or {})
    for run_id in (answers or {}):
        saved = stored.get(run_id)
        if not isinstance(saved, dict):
            continue
        entry = ledger.reels.setdefault(run_id, {"run_id": run_id})
        for field in PROVIDER_ANNOTATIONS:
            if saved.get(field):
                entry[field] = saved[field]

    # The same merge in the other direction, for the rows a provider *cleared*.
    # A reel whose post turns out to belong to another reel is disowned on
    # disk, and it does not appear in `answers` - it has no numbers, that is
    # the point - so the loop above never reaches it and the save below would
    # put the borrowed numbers straight back. Merging field by field cannot
    # express a deletion, so the disk row is adopted whole.
    for run_id in [row["run_id"] for row in waiting]:
        saved = stored.get(run_id)
        held = ledger.reels.get(run_id)
        if not isinstance(saved, dict) or not isinstance(held, dict):
            continue
        if held.get("media_id") and not saved.get("media_id"):
            ledger.reels[run_id] = saved

    recorded = 0
    for run_id, metrics in (answers or {}).items():
        if not isinstance(metrics, dict):
            continue
        problems = ledger.record(run_id, metrics, source="provider")
        if any("no usable numbers" in problem for problem in problems):
            warnings.extend(f"{run_id}: {problem}" for problem in problems)
            continue
        warnings.extend(f"{run_id}: {problem}" for problem in problems)
        recorded += 1
    warnings.extend(ledger.save())
    return recorded, warnings


# ---------------------------------------------------------------------------
# Printing
# ---------------------------------------------------------------------------

RULE = "-" * 78


def _hours_words(hours: float) -> str:
    """`6.73` as `6h44m`. Minutes, because the last hour of a window matters."""
    if hours < 0:
        hours = 0.0
    whole = int(hours)
    minutes = int(round((hours - whole) * 60))
    if minutes == 60:
        whole, minutes = whole + 1, 0
    if whole >= 24:
        days, spare = divmod(whole, 24)
        return f"{days}d{spare}h" if spare else f"{days}d"
    return f"{whole}h{minutes:02d}m"


def read_window(row: dict, now: datetime | None = None) -> dict:
    """When this reel should be read, and whether that moment has come.

    Every part of this was already in the ledger - when the reel went out, how
    old its last reading is, and the two constants that bound a comparable
    reading - and `unsettled` already selects on it. What nothing did was say
    it out loud, so read by hand the window had to be worked out in the head of
    whoever remembered to look. The two reels read since the insights loop went
    live were read at 18h and 26h: one comparable by luck and one not
    comparable at all. That is not a small loss - read at 18h,
    `why-some-trees` was reach 7; the same reel read again at 41h was reach 20.

    `state` is one of:

      unposted  nothing has recorded it as posted, so there is no clock yet
      early     too young to read, and `opens_in` hours from being worth it
      ripe      read it now; `shuts_in` hours before it stops counting
      missed    the window shut with no comparable reading, and will not reopen
      done      it already has a reading inside the window; nothing to do

    `done` and `missed` are both terminal, and they are different: one of them
    is a measurement and the other is a reel that can never be compared with
    the rest, however many times it is read afterwards.
    """
    now = now or datetime.now(timezone.utc)
    opens = float(config.PERFORMANCE_MIN_AGE_HOURS)
    shuts = float(config.PERFORMANCE_WINDOW_HOURS)
    blank = {"state": "", "age": None, "opens_in": None, "shuts_in": None,
             "opens_at": None, "shuts_at": None, "reading": None}

    reading = row.get("hours_after_post")
    reading = None if reading is None else float(reading)
    status = row.get("status", "pending")

    # A reading at or past the lower edge is the one the report will use and
    # `unsettled` retires the reel on it, so there is no window left to report.
    if status == "posted" and reading is not None and reading >= opens:
        return {**blank, "state": "done", "reading": reading}

    # An assumed posting time is the time the reel was *made*. Naming a due
    # hour off it would dress a guess up as a clock, and the clock is the whole
    # point of this - so a reel nothing has posted has no window, it has a
    # missing fact, and the line says which.
    posted = None
    if status == "posted" and not row.get("posted_at_assumed"):
        posted = _parse_stamp(row.get("posted_at"))
    if posted is None:
        return {**blank, "state": "unposted", "reading": reading}

    age = (now - posted).total_seconds() / 3600.0
    window = {
        "age": age,
        "reading": reading,
        "opens_at": posted + timedelta(hours=opens),
        "shuts_at": posted + timedelta(hours=shuts),
    }
    if age < opens:
        return {**window, "state": "early",
                "opens_in": opens - age, "shuts_in": shuts - age}
    if age <= shuts:
        return {**window, "state": "ripe",
                "opens_in": 0.0, "shuts_in": shuts - age}
    return {**window, "state": "missed", "opens_in": 0.0, "shuts_in": 0.0}


def read_due_line(row: dict, now: datetime | None = None) -> str:
    """`read_window` as the one line that goes under a reel on the screen."""
    return _due_words(read_window(row, now))


def _due_words(window: dict) -> str:
    """The line, from a window that has already been computed."""
    state = window["state"]
    reading = window["reading"]
    was = ("never read" if reading is None
           else f"read at {reading:g}h, too early")

    if state == "unposted":
        return "not recorded as posted - nothing is counting yet"
    if state == "early":
        return (f"{was} - ripe in {_hours_words(window['opens_in'])}, "
                f"then {_hours_words(window['shuts_in'] - window['opens_in'])} "
                f"to read it")
    if state == "ripe":
        return (f"{was} - READ NOW, the window shuts in "
                f"{_hours_words(window['shuts_in'])}")
    if state == "missed":
        return (f"{was} - the window shut "
                f"{_hours_words(window['age'] - float(config.PERFORMANCE_WINDOW_HOURS))}"
                f" ago; it can no longer be compared")
    if state == "done":
        return f"read at {reading:g}h - comparable, nothing to do"
    return ""


#: The order the screen puts them in. Not the order they were made: a window
#: that shuts today outranks one that opens tomorrow, and `missed` sits at the
#: bottom because nothing can be done about it.
_DUE_ORDER = {"ripe": 0, "early": 1, "unposted": 2, "missed": 3, "done": 4}


def _print_pending(ledger: PerformanceLedger, limit: int = 14) -> None:
    """What is waiting for numbers, when each is due, and the command for it.

    Listed off `unsettled` and not off `pending`, which is the fix here. Those
    two are different sets and the screen was showing the smaller one: a reel
    that was posted and read too early is still waiting for the reading that
    counts, `--fetch` asks about it, and the screen left it out entirely. Read
    on 2026-09-11 the difference was the whole story - one reel listed, and a
    second one six hours from the end of a window nothing had mentioned.
    """
    rows = ledger.rows()
    if not rows:
        print("no reels recorded yet.")
        print("  python pipeline/run.py     makes one, and registers it here")
        return

    now = datetime.now(timezone.utc)
    # Paired with its window once, because `read_window` reads the clock and a
    # row sorted by one call and printed by another can disagree with itself.
    due = [(row, read_window(row, now)) for row in ledger.unsettled()]
    due.sort(key=lambda pair: (_DUE_ORDER.get(pair[1]["state"], 9),
                               -(pair[1]["age"] or 0.0)))
    due = due[:limit]

    print(f"{len(rows)} reel(s) known: "
          f"{sum(1 for r in rows if r.get('status') == 'posted')} posted, "
          f"{sum(1 for r in rows if r.get('status') == 'skipped')} skipped, "
          f"{len(ledger.pending())} never recorded as posted.")
    if not due:
        print("every posted reel has a reading inside its window. "
              "python pipeline/perform.py --report")
        return

    ripe = sum(1 for _, window in due if window["state"] == "ripe")
    print(RULE)
    print(f"WAITING FOR NUMBERS   ({ripe} ready to read now)"
          if ripe else "WAITING FOR NUMBERS")
    for row, window in due:
        features = row.get("features") or {}
        print(f"  {features.get('day', '?'):<12} {row['run_id']}")
        print(f"    {features.get('topic') or '(no topic recorded)'}")
        print(f"    {_due_words(window)}")
    print(RULE)
    print(f"a reading counts only between {config.PERFORMANCE_MIN_AGE_HOURS}h "
          f"and {config.PERFORMANCE_WINDOW_HOURS}h after posting, and each reel "
          f"gets one shot at it")
    if config.PERFORMANCE_PROVIDER:
        print("  python pipeline/perform.py --fetch"
              "        reads every one of them off Instagram")
    example = due[-1][0]["run_id"].rsplit("-", 1)[-1]
    print("record how one did, or that it never went out:")
    print(f"  python pipeline/perform.py {example} --views 4200 --watched 61 --saves 30")
    print(f"  python pipeline/perform.py {example} --skipped footage"
          f"   ({', '.join(list(SKIP_REASONS)[:5])}, ...)")


def _log_warnings(path, limit: int = 3) -> list[str]:
    """The WARNINGS block of a run transcript, as single lines.

    Read from the log rather than from the ledger because the ledger keeps only
    the count. The block is the last thing a run prints and every line in it
    starts "  - ", so it is found by that shape rather than by position - a run
    that failed prints the same block from a different place in the file.
    """
    out: list[str] = []
    try:
        body = Path(path).read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return out
    seen = False
    for line in body.splitlines():
        if line.strip() == "WARNINGS":
            seen = True
            continue
        if not seen:
            continue
        stripped = line.strip()
        if stripped.startswith("- "):
            out.append(" ".join(stripped[2:].split()))
            if len(out) >= limit:
                break
        elif stripped.startswith("---") or not stripped:
            break
    return out


def _print_relabel(ledger: PerformanceLedger, limit: int = 20) -> None:
    """The binned reels with no reason, what survives of each, and the command.

    A worksheet rather than a prompt. The verdict on a reel is a judgement
    about something that was watched, and nothing here is in a position to make
    it - what this can do is put the evidence and the exact command on one
    screen so a backlog of them is one pass instead of nine hunts through
    output/logs/.
    """
    rows = unlabelled_bins(ledger.rows())
    if not rows:
        print("every binned reel carries a reason.")
        return

    print(f"{len(rows)} binned reel(s) have no reason on them.")
    print("They are read by neither table: `unlabelled` is a verdict the "
          "ledger failed to")
    print("keep, not a verdict about the reel. One word each puts them into "
          "the tally that")
    print("says what to go and fix.")
    if config.RUN_LOG_KEEP_DAYS:
        print()
        print(f"The evidence expires. Run directories are pruned and "
              f"output/logs/ keeps {config.RUN_LOG_KEEP_DAYS} days,")
        print("so a bin left unlabelled for longer than that cannot be "
              "settled later at all.")
    print(RULE)

    for row in rows[-limit:]:
        features = row.get("features") or {}
        run_id = row["run_id"]
        print(f"  {features.get('day', '?'):<12} {run_id}")
        print(f"    topic : {features.get('topic') or '(none recorded)'}")
        said = str(row.get("skip_note") or row.get("skip_reason") or "").strip()
        if said:
            print(f"    said  : \"{said}\"")
        video = str(features.get("video_path") or "")
        print(f"    reel  : {video if video and Path(video).is_file() else 'gone - the run directory was pruned'}")
        log = recover_log(run_id, features)
        if log:
            print(f"    log   : {log}")
            for warning in _log_warnings(log):
                print(f"      - {warning[:110]}")
        else:
            print("    log   : gone")
        print(f"    label : python pipeline/perform.py "
              f"{run_id.rsplit('-', 1)[-1]} --skipped <code>")
        print()

    if len(rows) > limit:
        print(f"  ... and {len(rows) - limit} older one(s).")
    print(RULE)
    print("  " + " | ".join(code for code in SKIP_REASONS
                            if code not in UNJUDGED_SKIPS))
    print("  python pipeline/perform.py --reasons     what each one means")


def _print_separation(readable: list[dict]) -> None:
    """What the rows above did and did not establish.

    A table that prints deltas and stops invites the largest one to be read as
    the finding, and that is exactly backwards when every interval in the
    column covers nought: the biggest delta in a table that has separated
    nothing is its noisiest row, not its most interesting one. It is the row
    with the least evidence under it, which is how it got to be the biggest.

    So the table says which of the two it is holding. "None of these separated"
    is a real result and a useful one - it is the report declining to hand over
    a finding it cannot support, in a project whose every other number is
    printed with the parts it was built from.
    """
    separated = [row for row in readable if row["separated"]]
    if separated:
        print(f"  * {len(separated)} of {len(readable)} row(s) separated from "
              f"nothing - the range clears zero. The rest sit inside it.")
        return
    widths = [(row["high"] - row["low"]) / 2.0 for row in readable
              if row["low"] is not None]
    if not widths:
        print("  no row here has two readings a side to resample; the deltas "
              "above are differences, not findings.")
        return
    biggest = max((abs(row["delta"]) for row in readable
                   if row["delta"] is not None), default=0.0)
    print(f"  none of the {len(readable)} readable row(s) separated from "
          f"nothing: every range above covers zero.")
    print(f"  at this many reels the table can only see a difference past "
          f"about {statistics.median(widths):.2f}, and the largest here is "
          f"{biggest:.2f}.")


def _print_table(title: str, rows: list[dict], *, unit: str,
                 min_samples: int, limit: int = 12) -> None:
    print(RULE)
    print(title)
    readable = [row for row in rows if row["readable"]]
    if not readable:
        print(f"  nothing has reached {min_samples} reels on both sides of a "
              f"question yet.")
        # Only the questions that actually have two sides are on their way to
        # being readable. A one-sided question printed here as "n=12 of 12"
        # reads as the closest of the lot when it is the furthest: every reel
        # answered it the same way, so the count is already final and the
        # other side is empty. Listing those in the same breath as the thin
        # ones is how "the pipeline never varies this" gets read as "four more
        # reels to go".
        contested = [row for row in rows if row.get("sides", 2) > 1]
        for row in sorted(contested, key=lambda entry: -entry["n"])[:4]:
            print(f"    {row['question']} = {row['answer']}: "
                  f"n={row['n']} of {row['of']}")
        if rows and not contested:
            print("    no question here has two answers to compare.")
        flat = one_sided(rows)
        if flat:
            print(f"  {len(flat)} question(s) had one answer on every reel and "
                  f"cannot be read at any n:")
            print("    " + ", ".join(flat[:6])
                  + (", ..." if len(flat) > 6 else ""))
        return
    label = f"{config.PERFORMANCE_CONFIDENCE:.0%} range"
    print(f"  {'question':<26} {'answer':<12} {'n':>3}  {'median':>8}  "
          f"{'vs rest':>8}  {label}")
    for row in readable[:limit]:
        delta = row["delta"]
        shown = f"{delta:+g}" if delta is not None else "-"
        span = ("-" if row["low"] is None
                else f"{row['low']:+g} .. {row['high']:+g}")
        print(f"  {row['question'][:26]:<26} {row['answer'][:12]:<12} "
              f"{row['n']:>3}  {row['median']:>7}{unit}  {shown:>8}  {span}"
              + ("  *" if row["separated"] else ""))
    _print_separation(readable)


def _charged(components: list, width: int = 65) -> str:
    """The component names that fit, cut at a comma and never mid-word.

    A name sliced in half - "curiosity (contradic" - reads as a component
    nobody recognises rather than as a line that ran out of room, and this
    column exists to be recognised.
    """
    if not components:
        return "nothing recorded"
    out = ""
    last = len(components) - 1
    for index, name in enumerate(components):
        nxt = f"{out}, {name}" if out else str(name)
        # Every name but the last has to leave room for the ", ..." that says
        # it was not the last, or the column overruns the width it declared and
        # the rule this report is drawn to.
        if len(nxt) > (width if index == last else width - 5):
            return f"{out}, ..." if out else str(name)[:width]
        out = nxt
    return out


def _print_reach_exhibit(exhibit: dict) -> None:
    """The reach distribution itself, printed because no table can hold it.

    Every other block in this report is a comparison and says so. This one is
    the observations, in the order they happened to land, and the heading says
    *that* - "an exhibit, not a finding" - because a list of components beside
    a list of big numbers is read as a claim about the components unless it is
    told not to be.
    """
    print(RULE)
    print("WHAT REACH ACTUALLY LOOKED LIKE  (every posted reel, no floors)")
    print("  an exhibit and not a finding - nothing here is read as an effect")
    listed = exhibit.get("rows") or []
    unread = exhibit.get("unread") or 0
    if not listed:
        print("  no posted reel has a reach reading on it yet"
              + (f" - {unread} waiting on python pipeline/instagram.py --fetch"
                 if unread else "."))
        return

    # The components go on their own line under each reel rather than in a
    # column, the way `_print_reasons` prints its notes: four component names
    # do not fit beside a topic inside the rule this report is drawn to, and
    # truncating them mid-word is how "curiosity" became "curi".
    print(f"  {'reach':>6} {'share':>5} {'score':>5} {'read':>5}  topic")
    for row in listed:
        share = f"{row['share']:.0%}" if row.get("share") is not None else "-"
        score = (f"{float(row['score']):.2f}"
                 if row.get("score") is not None else "-")
        hours = (f"{float(row['hours']):.0f}h"
                 if row.get("hours") is not None else "-")
        print(f"  {row['reach']:>6} {share:>5} {score:>5} {hours:>5}  "
              f"{(row.get('topic') or row['run_id'])[:44]}")
        # Indented like the notes under WHY THEY WERE BINNED rather than
        # aligned under the topic column: alignment cost thirteen characters
        # and thirteen characters was the difference between showing which
        # curiosity signal fired and showing "explainer shape, ...".
        print(f"             {_charged(row.get('components') or [])}")

    total = exhibit.get("total") or 0
    of = exhibit.get("of") or 0
    concentration = exhibit.get("concentration")
    if concentration is not None and of > 1:
        print(f"  the best reel holds {concentration:.0%} of all {total} "
              f"impression(s) across {of} reel(s)")
        if concentration > _NO_MIDDLE_SHARE:
            # Arithmetic, not a judgement: one reel outweighing all the others
            # put together drags the mean above every observation but itself.
            print(f"  that is more than all the others together: the mean "
                  f"sits above {of - 1} of the {of},")
            print(f"  so there is no ordinary reel here for the tables to "
                  f"describe")

    # The two orderings side by side, and nothing inferred from the pair. If
    # the board's favourites are also the ones that reached people, these two
    # lines agree and cost a glance; if they do not, that is the cheapest
    # possible look at a weight nothing in this report can otherwise read.
    by_score = exhibit.get("by_score") or []
    if by_score and of > 1:
        print("  ranked by the board's own score instead, the top "
              f"{len(by_score)} were:")
        for row in by_score:
            print(f"    {float(row['score']):.2f}  "
                  f"{(row.get('topic') or row['run_id'])[:44]:<44} "
                  f"reached {row['reach']}")
    if unread:
        print(f"  {unread} posted reel(s) have no reach reading and are not "
              f"listed - never asked is not nobody saw it")


def _print_reasons(rows: list[dict]) -> None:
    """Why the binned reels were binned. The column that arrives first."""
    print(RULE)
    print("WHY THEY WERE BINNED  (the same-day verdict; no API involved)")
    if not rows:
        print("  nothing binned carries a reason yet.")
        print("  python pipeline/perform.py <run> --skipped "
              + "|".join(list(SKIP_REASONS)[:4]) + "|...")
        return
    for row in rows:
        print(f"  {row['code']:<10} {row['n']:>3}  {row['share']:>5.0%}  "
              f"{row['means']}")
        for note in row["notes"]:
            print(f"             {note[:58]}")


def _print_audience(audience: dict, reach_median, own: int = 0,
                    real: dict | None = None) -> None:
    """The account's own size and what strangers did, before anything is
    divided by either.

    Every ratio below this line is a reel measured against the people who were
    shown it, and for the first eight posts that was 289, 64, 9, 6, 3, 3, 0 and
    0. The report printed all of those as `1.00x an ordinary reel` and never
    said how many accounts were behind them.

    The second half is the one the ratio tables structurally cannot produce.
    They compare reels against each other, so a column holding the same
    constant on every row is not a finding to them - and likes here were 3, 3,
    3, 3, 3, 3, 3 and 4. Read as a total that is twenty-five likes and a quiet
    account; read against PERFORMANCE_OWN_ACCOUNTS it is **one like, ever**,
    which is a different fact and the one worth printing.

    Silent about the count when nothing has been read, rather than printing a
    zero: an account with no followers and a token that cannot see the follower
    count are different findings, and only one of them is about the reels.
    """
    real = real or {}
    followers = audience.get("followers")

    if followers is None:
        print("AUDIENCE: not read - "
              "python pipeline/instagram.py --fetch records it, and the "
              "reach figures below have no denominator without it")
    else:
        line = f"AUDIENCE: {followers} follower(s)"
        if audience.get("username"):
            line += f" on @{audience['username']}"
        if audience.get("posts") is not None:
            line += f", {audience['posts']} post(s)"
        # Said plainly rather than left to be worked out. A follower count that
        # is entirely the operator's own logins is not a small audience, it is
        # no audience, and the two lead to opposite decisions.
        if own > 0 and followers <= own:
            line += " - your own accounts, so the audience here is nobody"
        elif reach_median is not None and followers > 0:
            # Above 1.00 the account is being shown to people who do not follow
            # it, which for this format is the only way it grows.
            line += (f" - a median reel reached {reach_median:g}, "
                     f"{reach_median / followers:.2f}x that")
        print(line)
        if audience.get("read_at"):
            print(f"          read {audience['read_at'][:10]}; it is today's "
                  f"figure against reels posted over weeks")

    if not own or not real.get("reels"):
        return

    seen = real.get("reach") or 0
    unseen = real.get("unseen") or 0
    print(f"REACHED : {seen} impression(s) that were not yours, across "
          f"{real['reels']} posted reel(s)"
          + (f"; {unseen} of them reached nobody at all" if unseen else ""))

    # Singular where the count is one: "1 likes, 0 comments" reads as a
    # rounding artefact, and this number is the point of the line.
    acts = [f"{real.get(name) or 0} {name[:-1] if real.get(name) == 1 else name}"
            for name in ("likes", "comments", "saves", "shares")]
    line = f"ACTED   : {', '.join(acts)}"
    if seen > 0:
        # The rate that matters and the one no table prints: of the people who
        # were actually shown a reel, how many did anything about it.
        line += f" - {(real.get('likes') or 0) / seen:.2%} of them liked it"
    print(line)

    # The line the report did not have. Said separately from ACTED because it
    # is a different question - that line is what viewers did about a reel,
    # this is what they did about the account - and because it has a third
    # state those metrics do not: never asked for. A zero here and a blank
    # here lead to opposite decisions.
    if not real.get("follow_reels"):
        print("FOLLOWED: not read on any reel - "
              "python pipeline/instagram.py --fetch now asks for follows and "
              "profile_visits; this stays blank if your token serves them "
              "only account-wide")
    else:
        follows = real.get("follows") or 0
        visits = real.get("profile_visits") or 0
        line = (f"FOLLOWED: {follows} follow(s) and {visits} profile visit(s) "
                f"across {real['follow_reels']} reel(s) read for them")
        if seen > 0:
            line += f" - {follows / seen:.2%} of those shown a reel followed"
        print(line)
    print(f"          {own} own account(s) are taken out first "
          f"(PERFORMANCE_OWN_ACCOUNTS); a reel that did not clear that "
          f"reached nobody, so nothing it carries is counted")


def _print_report(data: dict) -> None:
    print(RULE)
    print(f"REELS   : {data['reels']} known - {data['posted']} posted, "
          f"{data['skipped']} skipped, {data['pending']} waiting")
    print(f"METRIC  : {data['metric']}, as a ratio to the median of the "
          f"{data['baseline_runs']} nearest reels in time")
    print(f"READS   : {data['compared']} comparable, "
          f"nothing below n={data['min_samples']} is read, "
          f"nothing measured outside {data['min_age_hours']}-"
          f"{data['window_hours']}h after posting"
          + (f", nothing under {data['min_reach']} reach"
             if data.get("min_reach") else ""))
    _print_audience(data.get("audience") or {}, data.get("reach_median"),
                    data.get("own_accounts") or 0, data.get("real") or {})

    _print_reasons(data.get("reasons") or [])
    _print_table("KEPT OR BINNED  (1.00 = always posted; every reel counts)",
                 data["kept"], unit="", min_samples=data["min_samples"])
    # Printed first of the three outcome tables, because it is upstream of the
    # other two: a reel nobody was shown cannot be watched through or saved,
    # and for this account that is the case that keeps happening.
    _print_table(
        f"HOW MANY SAW IT  (1.00 = an ordinary reach for this account"
        + (f"; median {data['reach_median']:g} accounts)"
           if data.get("reach_median") is not None else ")"),
        data.get("reached") or [], unit="x", min_samples=data["min_samples"])
    # Directly under the table that asks the same question and cannot answer
    # it. When HOW MANY SAW IT prints "nothing has reached n reels on both
    # sides", this is what it was unable to say anything about, and leaving a
    # reader to go and read the ledger for it is how the account's actual
    # shape stayed unprinted for eight reels.
    _print_reach_exhibit(data.get("reach_exhibit") or {})
    _print_table(f"HOW THEY DID  (1.00 = an ordinary reel for this account)",
                 data["performance"], unit="x", min_samples=data["min_samples"])
    _print_table("SAVED OR SENT  (1.00 = every reel got a save or a share)",
                 data["saved"], unit="", min_samples=data["min_samples"])
    # Printed last because it is the last thing that happens, and the only
    # outcome here that survives the reel. Everything above it is attention
    # paid once; a follower is attention that comes back.
    _print_table(
        "FOLLOWED AFTER  (1.00 = every reel won a follow or a profile visit)",
        data.get("followed") or [], unit="", min_samples=data["min_samples"])

    print(RULE)
    if data["warnings"]:
        print("warnings:")
        for warning in data["warnings"]:
            print(f"  - {warning}")
        print(RULE)
    print("nothing here is applied to the scoring. Move a weight in "
          "pipeline/topics.py yourself,")
    print("or set PERFORMANCE_FEEDBACK=true to let a bounded 'measured' "
          "component onto the board.")
    print(RULE)


def _print_recorded(run_id: str, entry: dict) -> None:
    features = entry.get("features") or {}
    print(f"{run_id}")
    if features.get("topic"):
        print(f"  {features['topic']}")
    if entry.get("status") == "skipped":
        code = entry.get("skip_code") or UNLABELLED_SKIP
        means = SKIP_REASONS.get(code, "no reason recorded")
        print(f"  skipped: {code} - {means}")
        if entry.get("skip_note"):
            print(f"           {entry['skip_note']}")
        if code == UNLABELLED_SKIP:
            print("  no code on it, so neither table reads it. One word "
                  "fixes that:")
            print(f"    python pipeline/perform.py {run_id} --skipped "
                  + "|".join(list(SKIP_REASONS)[:4]) + "|...")
            return
        print("  kept as evidence - a reel you did not post is the fastest "
              "quality signal there is.")
        return
    metrics = entry.get("metrics") or {}
    print("  " + ", ".join(
        f"{name.replace('_', ' ')} "
        + (f"{value:.0%}" if name in RATE_METRICS
           else f"{value:g}" if name in FLOAT_METRICS else f"{value:,}")
        for name, value in metrics.items()
    ))
    if entry.get("hours_after_post") is not None:
        age = entry["hours_after_post"]
        print(f"  measured {age:g}h after posting"
              + ("  (posting time assumed - pass --posted to correct it)"
                 if entry.get("posted_at_assumed") else ""))
        if age < config.PERFORMANCE_MIN_AGE_HOURS:
            print(f"  too green to compare with anything until "
                  f"{config.PERFORMANCE_MIN_AGE_HOURS}h - --fetch reads it "
                  f"again")
    readings = entry.get("measurements") or []
    if len(readings) > 1:
        trail = ", ".join(
            f"{(reading.get('hours_after_post') or 0):g}h "
            f"{(reading.get('metrics') or {}).get('views', '-')}"
            for reading in readings[-4:]
        )
        print(f"  views as it aged: {trail}")
    if saturated(entry):
        print("  watch-through is on its cap, so it is ranked on watch_ratio "
              f"({(metrics.get('watch_ratio', 0)):g}x its own length)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
# Writing it down
#
# The reel is made by hand and posted by hand on purpose. Everything between
# those two was supposed to happen unattended, and the reading was - while the
# thing the reading is *for* still required somebody to remember to type
# `--report`. A number nobody looks at is the same as a number nobody took.
# ---------------------------------------------------------------------------

def render_report(data: dict) -> str:
    """The report exactly as `--report` prints it, as a string.

    Captured from `_print_report` rather than rebuilt beside it. Two renderers
    drift: the file would keep printing a table the terminal had stopped
    showing, and the first person to notice would be someone quoting a number
    that no longer exists. There is one report and this is a second way of
    spelling it.
    """
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        _print_report(data)
    return buffer.getvalue()


def write_report(data: dict, *, path: Path | None = None,
                 json_path: Path | None = None,
                 history_dir: Path | None = None,
                 keep: int | None = None,
                 today: date | None = None) -> list[str]:
    """Write the report to disk. Never raises - a failed write is a warning.

    Deliberately not fatal. This runs at the tail of the daily read, and the
    ledger has already been saved by the time it does: a full disk or a file
    open in an editor must not turn a successful read into a failed task and
    cost the day's numbers. The reading is the job; this is the copy.

    Three files, because they answer different questions. `report.txt` is
    always the latest and never has to be found. `report.json` is the same
    reading for anything that wants the numbers rather than the table. The
    dated copy under `reports/` is how a claim about what the report said last
    week gets checked rather than remembered - which matters here more than it
    usually would, because a reel read at 19h and re-read at 43h produces two
    honestly different reports and only the second one is comparable.
    """
    path = Path(path or config.PERFORMANCE_REPORT_FILE)
    json_path = Path(json_path or config.PERFORMANCE_REPORT_JSON)
    history_dir = Path(history_dir or config.PERFORMANCE_REPORT_DIR)
    keep = config.PERFORMANCE_REPORT_HISTORY if keep is None else keep
    stamp = (today or datetime.now(timezone.utc).date()).isoformat()

    warnings: list[str] = []
    try:
        text = render_report(data)
    except Exception as exc:  # noqa: BLE001 - a report is never worth a crash
        return [f"the report could not be rendered ({exc})"]

    header = (f"# {stamp} - written by the daily read. "
              f"python pipeline/perform.py --report prints this live.\n\n")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header + text, encoding="utf-8")
    except OSError as exc:
        warnings.append(f"the report could not be written to {path} ({exc})")

    try:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                             encoding="utf-8")
    except OSError as exc:
        warnings.append(f"the report json could not be written ({exc})")

    # The dated copy, and then the pruning. Written after the two above so a
    # directory that cannot be made costs only the history and not the report.
    try:
        history_dir.mkdir(parents=True, exist_ok=True)
        (history_dir / f"{stamp}.txt").write_text(header + text,
                                                  encoding="utf-8")
    except OSError as exc:
        warnings.append(f"the dated report copy could not be written ({exc})")
        return warnings

    if keep > 0:
        try:
            dated = sorted(history_dir.glob("????-??-??.txt"))
            for stale in dated[:-keep]:
                stale.unlink()
        except OSError as exc:
            warnings.append(f"old report copies could not be pruned ({exc})")
    return warnings


def assume_audio(ledger: PerformanceLedger, mode: str = "") -> list[str]:
    """Fill in the audio of posted reels nobody is going to type it for.

    Off unless `PERFORMANCE_ASSUME_AUDIO` says otherwise, and that is the right
    default for anyone who has not said: `note_audio` exists *because* the
    pipeline cannot see what happened in the Instagram app, and inventing an
    answer is the failure it was built to prevent. Set by an operator who posts
    the rendered file as-is and never touches audio in the app, it is not an
    inference - it is that operator stating a fact about their own workflow
    once instead of fourteen times.

    Two things it will not do, and both are the reason it is safe to leave on:

      * **it never overwrites.** An audio recorded by hand outranks a standing
        assumption, always. That is what makes "I usually don't, but this once
        I did" recoverable - record the exception and this leaves it alone.
      * **it never assumes a bed onto a reel that had none.** `features`
        knows what step 5 rendered. Where that is no bed, the only two truths
        are `silent` and `added`, this cannot tell them apart, and the one it
        would guess is the one that inverts the `music` trial. So it declines.
    """
    mode = (mode or config.PERFORMANCE_ASSUME_AUDIO or "").strip().lower()
    if not mode:
        return []
    if mode not in AUDIO_MODES:
        return [f"PERFORMANCE_ASSUME_AUDIO={mode!r} is not one of "
                + ", ".join(sorted(AUDIO_MODES)) + " - nothing was assumed"]

    filled, declined = 0, 0
    for row in ledger.rows():
        if row.get("status") != "posted" or row.get("audio"):
            continue
        rendered = ((row.get("features") or {}).get("music") or "")
        if mode == "bed" and not rendered:
            declined += 1
            continue
        ledger.note_audio(row.get("run_id") or "", mode)
        filled += 1

    warnings: list[str] = []
    if filled:
        warnings.append(
            f"recorded audio={mode} on {filled} posted reel(s) from "
            f"PERFORMANCE_ASSUME_AUDIO - set it to \"\" to be asked instead"
        )
    if declined:
        warnings.append(
            f"{declined} posted reel(s) were rendered with no music bed, so "
            f"audio={mode} was not assumed onto them - only you know whether "
            f"a track was added in the app. python pipeline/perform.py <run> "
            f"--audio silent|added"
        )
    return warnings


def refresh_outputs(*, metric: str = "",
                    ledger: PerformanceLedger | None = None) -> list[str]:
    """Everything the daily read does once the numbers are in. Never raises.

    Called from two places that both read insights and would otherwise disagree
    about what happens next: `--fetch`, which is what the read task runs, and
    step 8's own read at the start of a manual run. Before this existed only
    the first of them wrote the report, so making a reel by hand refreshed the
    ledger and left `report.txt` describing the ledger as it had been - the one
    failure mode that looks exactly like a working report.

    Reads a fresh ledger by default, because `fetch` has already saved and the
    caller may not be holding the copy it saved.
    """
    ledger = ledger or PerformanceLedger()
    warnings = list(assume_audio(ledger))
    if warnings:
        warnings.extend(ledger.save())
    if not config.PERFORMANCE_REPORT_ON_FETCH:
        return warnings
    try:
        data = report(ledger=ledger, metric=metric)
        warnings.extend(write_report(data))
    except Exception as exc:  # noqa: BLE001 - the read is the job, not this
        warnings.append(
            f"the read succeeded and the report did not ({exc}) - "
            f"python pipeline/perform.py --report"
        )
    return warnings


# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Record how a reel did, and read back what that says "
                    "about how they are being made (step 9).",
    )
    parser.add_argument("run", nargs="?", default=None,
                        help="the reel: its run id, the short hash on the end "
                             "of it, its directory, or \"latest\"")
    parser.add_argument("--views", type=float, default=None)
    parser.add_argument("--reach", type=float, default=None)
    parser.add_argument("--watched", type=float, default=None, dest="watch_through",
                        metavar="PCT",
                        help="watch-through, as a percentage or a fraction")
    parser.add_argument("--watch-seconds", type=float, default=None,
                        dest="avg_watch_seconds",
                        help="average watch time in seconds")
    parser.add_argument("--likes", type=float, default=None)
    parser.add_argument("--comments", type=float, default=None)
    parser.add_argument("--saves", type=float, default=None)
    parser.add_argument("--shares", type=float, default=None)
    parser.add_argument("--follows", type=float, default=None)
    parser.add_argument("--profile-visits", type=float, default=None,
                        dest="profile_visits",
                        help="viewers who opened the profile from this reel")
    parser.add_argument("--posted", default="", metavar="WHEN",
                        help="when it went out (YYYY-MM-DD). Without it the "
                             "age of the measurement is taken from when the "
                             "reel was made")
    parser.add_argument("--audio", default="", metavar="MODE",
                        choices=("", *AUDIO_MODES),
                        help="what was actually playing under the reel when "
                             "it went out: bed (the rendered bed), silent "
                             "(nothing), or added (a track attached in the "
                             "Instagram app - name it with --track). Step 5 "
                             "cannot know this: Instagram's music catalogue "
                             "has no publishing API")
    parser.add_argument("--track", default="", metavar="NAME",
                        help="which track was attached, for --audio added. "
                             "Kept on the row but never faceted on - one "
                             "track per reel is a bucket per reel")
    parser.add_argument("--skipped", nargs="?", const="", default=None,
                        metavar="WHY",
                        help="this reel was made and never posted, and why: "
                             + ", ".join(SKIP_REASONS)
                             + ". Anything else is kept as a note and filed "
                               "as " + UNLABELLED_SKIP)
    parser.add_argument("--note", default="", metavar="TEXT",
                        help="the sentence behind the code, kept alongside it")
    parser.add_argument("--reasons", action="store_true",
                        help="the skip vocabulary, and what each code means")
    parser.add_argument("--relabel", action="store_true",
                        help="the binned reels whose reason was lost, what "
                             "evidence survives of each, and the command to "
                             "label it")
    parser.add_argument("--report", action="store_true",
                        help="what the ledger has learned so far")
    parser.add_argument("--fetch", action="store_true",
                        help="ask PERFORMANCE_PROVIDER about every reel still "
                             "waiting for numbers")
    parser.add_argument("--metric", default="",
                        help=f"metric to report on (default: "
                             f"{config.PERFORMANCE_METRIC})")
    parser.add_argument("--limit", type=int, default=14, metavar="N",
                        help="reels to list (default: 14)")
    parser.add_argument("--json", action="store_true",
                        help="print the report, or the entry, as JSON")
    args = parser.parse_args(argv)

    ledger = PerformanceLedger()

    # Before anything is read or written: reels made before this step existed
    # are the only history there is, and they are sitting in output/.
    adopted, problems = backfill(ledger)
    for problem in problems:
        print(f"warning: {problem}", file=sys.stderr)
    if adopted and not args.json:
        print(f"adopted {adopted} reel(s) already in output/ - what step 0 "
              f"scored them on was not kept, so those rows carry the reel but "
              f"not its components.")

    if ledger.migrated:
        # Written back rather than re-derived on every read, so the file on
        # disk is the upgraded one - the same trade backfill already makes.
        for problem in ledger.save():
            print(f"warning: {problem}", file=sys.stderr)
        if not args.json:
            print(f"brought {ledger.migrated} row(s) up to ledger version "
                  f"{LEDGER_VERSION} - watch_ratio derived from what they "
                  f"already held, and skips written before the vocabulary are "
                  f"{UNLABELLED_SKIP} until you say otherwise.")

    try:
        if args.reasons:
            print("why a reel was made and never posted:")
            for code, means in SKIP_REASONS.items():
                print(f"  {code:<10} {means}")
            print(f"\n  python pipeline/perform.py <run> --skipped footage")
            print(f"  python pipeline/perform.py <run> --skipped hook "
                  f"--note \"the card gave the answer away\"")
            print("  python pipeline/perform.py --relabel"
                  "        the bins still missing one")
            return 0

        if args.relabel:
            _print_relabel(ledger, max(1, args.limit))
            return 0

        if args.fetch:
            recorded, warnings = fetch(ledger=ledger)
            print(f"recorded {recorded} reel(s) from PERFORMANCE_PROVIDER.")

            # The whole point of the daily read: it writes down what it
            # found, so the report exists whether or not anybody types
            # `--report`. Shared with step 8's read so the two cannot drift.
            warnings.extend(refresh_outputs(metric=args.metric, ledger=ledger))
            if config.PERFORMANCE_REPORT_ON_FETCH:
                print(f"wrote {config.PERFORMANCE_REPORT_FILE}")

            for warning in warnings:
                print(f"warning: {warning}", file=sys.stderr)
            return 0

        if args.report or (args.run is None and args.json):
            data = report(ledger=ledger, metric=args.metric)
            print(json.dumps(data, indent=2, ensure_ascii=False)
                  if args.json else "", end="")
            if not args.json:
                _print_report(data)
            return 0

        if args.run is None:
            _print_pending(ledger, max(1, args.limit))
            return 0

        run_id = resolve_run(args.run, ledger=ledger)

        # Ahead of the metrics branch, and deliberately not part of it. The
        # audio is known the moment the reel is posted; the numbers are not
        # known for hours. Requiring them together would mean either recording
        # the audio from memory days later or not at all, and this is a fact
        # that decays - nobody remembers in a week which of fourteen reels had
        # a track put on it.
        audio_recorded = False
        if args.audio:
            for warning in ledger.note_audio(run_id, args.audio, args.track):
                print(f"warning: {warning}", file=sys.stderr)
            audio_recorded = True

        if args.skipped is not None:
            for warning in ledger.skip(run_id, args.skipped):
                print(f"warning: {warning}", file=sys.stderr)
            if args.note:
                entry = ledger.entry(run_id)
                entry["skip_note"] = args.note.strip()
                entry["skip_reason"] = args.note.strip()
        else:
            metrics = {name: getattr(args, name) for name in METRICS
                       if getattr(args, name, None) is not None}
            if not metrics and not audio_recorded:
                # Naming a reel and passing no numbers is a question, not a
                # record: show what is already known about it.
                entry = ledger.entry(run_id)
                if args.json:
                    print(json.dumps(entry, indent=2, ensure_ascii=False))
                else:
                    _print_recorded(run_id, entry)
                return 0
            if not metrics:
                # Audio alone is a perfectly good thing to record, and saying
                # so is not the same as having nothing to say.
                for warning in ledger.save():
                    print(f"warning: {warning}", file=sys.stderr)
                entry = ledger.entry(run_id)
                if args.json:
                    print(json.dumps(entry, indent=2, ensure_ascii=False))
                else:
                    _print_recorded(run_id, entry)
                return 0
            for warning in ledger.record(run_id, metrics, posted_at=args.posted):
                print(f"warning: {warning}", file=sys.stderr)

        for warning in ledger.save():
            print(f"warning: {warning}", file=sys.stderr)
    except PerformanceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    entry = ledger.entry(run_id)
    if args.json:
        print(json.dumps(entry, indent=2, ensure_ascii=False))
    else:
        _print_recorded(run_id, entry)
        remaining = len(ledger.pending())
        if remaining:
            print(f"\n{remaining} reel(s) still waiting. "
                  f"python pipeline/perform.py --report")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
