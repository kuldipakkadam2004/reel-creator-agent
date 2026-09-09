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
import json
import statistics
import sys
from datetime import date, datetime, timezone
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
FEATURES_VERSION = 5

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
           "follows")

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
}


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


def _segment(rows: list[dict], score, vocabulary: dict[str, list[str]],
             min_samples: int, statistic=statistics.median) -> list[dict]:
    """Group reels by facet and compare each answer against the rest.

    `score` maps a row to a number, or to None where the row cannot be scored.
    `statistic` summarises a bucket: the median for performance, where one
    freak reel should not carry a segment, and the mean for a rate like "how
    often did this get posted", where the median of a column of ones and zeros
    is a majority vote rather than a share.
    """
    buckets: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        value = score(row)
        if value is None:
            continue
        for pair in facets(row.get("features") or {}, vocabulary):
            buckets.setdefault(pair, []).append(value)

    questions: dict[str, list[str]] = {}
    for question, answer in buckets:
        questions.setdefault(question, []).append(answer)

    out: list[dict] = []
    for question, answers in questions.items():
        for answer in sorted(set(answers)):
            mine = buckets[(question, answer)]
            rest = [value for other in set(answers) if other != answer
                    for value in buckets[(question, other)]]
            here = statistic(mine)
            there = statistic(rest) if rest else None
            out.append({
                "question": question,
                "answer": answer,
                "n": len(mine),
                "median": round(here, 3),
                "against": round(there, 3) if there is not None else None,
                "delta": round(here - there, 3) if there is not None else None,
                "readable": len(mine) >= min_samples and len(rest) >= min_samples,
                "of": len(mine) + len(rest),
            })
    out.sort(key=lambda entry: (not entry["readable"],
                                -abs(entry["delta"] or 0.0), entry["question"]))
    return out


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

    total = len(judged)
    out = [{
        "code": code,
        "means": SKIP_REASONS.get(code, code),
        "n": count,
        "share": round(count / total, 3) if total else 0.0,
        "notes": notes.get(code, [])[-3:],
    } for code, count in counts.items()]
    out.sort(key=lambda row: (-row["n"], row["code"]))
    return out


def report(*, ledger: PerformanceLedger | None = None, metric: str = "",
           span: int | None = None, min_samples: int | None = None,
           window_hours: int | None = None,
           min_age_hours: int | None = None) -> dict:
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
        vocabulary, floor,
    )

    # Read on the same window as the performance table, so every reel is
    # judged at a comparable age: a reel read at 96h has had three more days to
    # be saved than one read at 29h, and a rate mixing the two measures how
    # long ago a reel went out as much as anything about the reel.
    saved = _segment(
        fresh, was_kept, vocabulary, floor, statistic=statistics.fmean,
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

    return {
        "metric": metric,
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
    with the number because it is what says whether to believe it, and a board
    that printed the effect without it would be the one dishonest line on an
    otherwise self-auditing page.

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
        if entry["readable"] and entry["answer"] == "fired"
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
    stored = _load(ledger.path).get("reels") or {}
    for run_id in (answers or {}):
        saved = stored.get(run_id)
        if not isinstance(saved, dict):
            continue
        entry = ledger.reels.setdefault(run_id, {"run_id": run_id})
        for field in PROVIDER_ANNOTATIONS:
            if saved.get(field):
                entry[field] = saved[field]

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


def _print_pending(ledger: PerformanceLedger, limit: int = 14) -> None:
    """What is waiting for numbers, and the command that records each one."""
    rows = ledger.rows()
    if not rows:
        print("no reels recorded yet.")
        print("  python pipeline/run.py     makes one, and registers it here")
        return

    waiting = ledger.pending()[-limit:]
    print(f"{len(rows)} reel(s) known: "
          f"{sum(1 for r in rows if r.get('status') == 'posted')} posted, "
          f"{sum(1 for r in rows if r.get('status') == 'skipped')} skipped, "
          f"{len(ledger.pending())} waiting.")
    if not waiting:
        print("nothing is waiting. python pipeline/perform.py --report")
        return

    print(RULE)
    print("WAITING FOR NUMBERS")
    for row in waiting:
        features = row.get("features") or {}
        print(f"  {features.get('day', '?'):<12} {row['run_id']}")
        print(f"    {features.get('topic') or '(no topic recorded)'}")
    print(RULE)
    example = waiting[-1]["run_id"].rsplit("-", 1)[-1]
    print("record how one did, or that it never went out:")
    print(f"  python pipeline/perform.py {example} --views 4200 --watched 61 --saves 30")
    print(f"  python pipeline/perform.py {example} --skipped footage"
          f"   ({', '.join(list(SKIP_REASONS)[:5])}, ...)")


def _print_table(title: str, rows: list[dict], *, unit: str,
                 min_samples: int, limit: int = 12) -> None:
    print(RULE)
    print(title)
    readable = [row for row in rows if row["readable"]]
    if not readable:
        print(f"  nothing has reached {min_samples} reels on both sides of a "
              f"question yet.")
        # What is closest to being readable, rather than whatever sorted first
        # among a field of ties.
        for row in sorted(rows, key=lambda entry: -entry["n"])[:4]:
            print(f"    {row['question']} = {row['answer']}: "
                  f"n={row['n']} of {row['of']}")
        return
    print(f"  {'question':<26} {'answer':<12} {'n':>3}  {'median':>8}  vs rest")
    for row in readable[:limit]:
        delta = row["delta"]
        print(f"  {row['question'][:26]:<26} {row['answer'][:12]:<12} "
              f"{row['n']:>3}  {row['median']:>7}{unit}  "
              + (f"{delta:+g}" if delta is not None else "-"))


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


def _print_report(data: dict) -> None:
    print(RULE)
    print(f"REELS   : {data['reels']} known - {data['posted']} posted, "
          f"{data['skipped']} skipped, {data['pending']} waiting")
    print(f"METRIC  : {data['metric']}, as a ratio to the median of the "
          f"{data['baseline_runs']} nearest reels in time")
    print(f"READS   : {data['compared']} comparable, "
          f"nothing below n={data['min_samples']} is read, "
          f"nothing measured outside {data['min_age_hours']}-"
          f"{data['window_hours']}h after posting")

    _print_reasons(data.get("reasons") or [])
    _print_table("KEPT OR BINNED  (1.00 = always posted; every reel counts)",
                 data["kept"], unit="", min_samples=data["min_samples"])
    _print_table(f"HOW THEY DID  (1.00 = an ordinary reel for this account)",
                 data["performance"], unit="x", min_samples=data["min_samples"])
    _print_table("SAVED OR SENT  (1.00 = every reel got a save or a share)",
                 data["saved"], unit="", min_samples=data["min_samples"])

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
    parser.add_argument("--posted", default="", metavar="WHEN",
                        help="when it went out (YYYY-MM-DD). Without it the "
                             "age of the measurement is taken from when the "
                             "reel was made")
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
            return 0

        if args.fetch:
            recorded, warnings = fetch(ledger=ledger)
            print(f"recorded {recorded} reel(s) from PERFORMANCE_PROVIDER.")
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
            if not metrics:
                # Naming a reel and passing no numbers is a question, not a
                # record: show what is already known about it.
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
