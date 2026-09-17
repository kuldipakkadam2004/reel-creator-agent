"""Step 8 - the end-to-end run: no topic in, finished reel out.

Chains steps 0, 2, 3, 4, 5 and 6 into one call. It is deliberately thin, and
that is the point: every step already returns a dict the next step accepts, so
there is no plumbing to invent here. What this module adds is the two jobs a
human was doing in between those commands - choosing which candidate off the
board to make, and deciding what happens when a step fails - plus the one thing
an unattended run needs that an interactive one never did: a trace.

Run by a scheduler at 5pm there is nobody watching stdout, so:
  * every line the six steps print is captured to output/logs/<stamp>.log
  * one summary per run is appended to output/runs.json
  * the exit code is 0 for a reel (or a deliberate skip) and 1 for a failure,
    which is what the OS scheduler reads to decide whether to retry

What it will not do is post anything. The reel and its caption land on disk and
you publish them yourself, exactly as before - this replaces the typing, not
the judgement about what goes out.

Standalone use:
    python pipeline/run.py                      # scout, choose, make it
    python pipeline/run.py "why the ocean is salty" -d 40
    python pipeline/run.py --dry-run            # choose a topic, make nothing
    python pipeline/run.py --status             # how the last runs went

To have it happen by itself every day, see pipeline/schedule.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date, datetime, timezone
from pathlib import Path

# Allow both `python pipeline/run.py` and `python -m pipeline.run`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from pipeline import (assemble, caption, perform, script_gen, topics, tts,  # noqa: E402
                      visuals)

# ---------------------------------------------------------------------------
# Tunables
# ---------------------------------------------------------------------------

#: Bumped when the shape of a ledger entry changes, so an old runs.json is
#: recognisable rather than silently misread.
RUN_VERSION = 1

#: The six steps in the order they run, for the progress lines. Step 1 is
#: config and step 7 is not built, so neither is a step this loop performs.
STEPS = ("topics", "script", "tts", "visuals", "assemble", "caption")

#: Failures that mean "this topic cannot be made" rather than "this pipeline is
#: broken". Only these fall through to the next candidate on the board, because
#: only these have any reason to come out differently on a different subject -
#: a rejected API key or a missing ffmpeg would fail identically on all ten,
#: having paid for each one to find out.
_TOPIC_FAILURES = (
    visuals.NoAssetsError,             # step 4 found no footage for these words
    assemble.NoFootageError,           # what step 4 found, step 5 cannot cut
    script_gen.ScriptGenerationError,  # nothing usable came back for the topic
)

#: Known ways for a run to end. Anything outside this list is a bug and is
#: allowed to propagate with its traceback intact, which is a distinction worth
#: keeping: these are failures the pipeline understands, and a traceback is not.
_RUN_FAILURES = (
    config.MissingConfigError,
    topics.TopicsError,
    script_gen.ScriptGenerationError,
    tts.TTSError,
    visuals.VisualsError,
    assemble.AssemblyError,
    caption.CaptionError,
    ValueError,
    OSError,
)

RULE = "-" * 68

#: Longest a `reason` may be before it is cut. It is one column of a ledger and
#: one line of a summary, not the error report - see `_one_line`.
_REASON_MAX_CHARS = 160


# ---------------------------------------------------------------------------
# Console + log capture
#
# The six step modules print. None of them logs, and teaching six modules to
# log so that one caller can keep a transcript is the wrong way round: the
# transcript is this module's problem, so this module solves it by sitting
# under their stdout for the length of the run.
# ---------------------------------------------------------------------------

class _Tee:
    """Write to the console and the log file at once.

    Tolerates a console that is not there. Under pythonw.exe - which is how a
    scheduled task runs without flashing a window onto the desktop every
    evening - sys.stdout is None, and a bare print() into that raises. The log
    file is the stream that actually matters, so a missing or broken console is
    written off rather than allowed to end the run.
    """

    def __init__(self, stream, handle):
        self._stream = stream
        self._handle = handle

    def write(self, text: str) -> int:
        if self._stream is not None:
            try:
                self._stream.write(text)
            except (ValueError, OSError, UnicodeEncodeError):
                pass
        self._handle.write(text)
        return len(text)

    def flush(self) -> None:
        for target in (self._stream, self._handle):
            try:
                target.flush()
            except (AttributeError, ValueError, OSError):
                pass

    def isatty(self) -> bool:
        try:
            return bool(self._stream is not None and self._stream.isatty())
        except (AttributeError, ValueError):
            return False

    def fileno(self) -> int:
        if self._stream is None:
            raise OSError("this run has no console attached")
        return self._stream.fileno()


class _Null:
    """Somewhere to print when there is neither a console nor a log.

    pythonw.exe gives a process no stdout at all, and `print()` into None
    raises. A run started that way with --no-log has genuinely nowhere to put
    its output, so it is dropped - but being dropped must not be the thing that
    ends the run, which is what a bare None does.
    """

    def write(self, text: str) -> int:
        return len(text)

    def flush(self) -> None:
        pass

    def isatty(self) -> bool:
        return False


def _attach_console() -> None:
    """Guarantee something to print to, before anything prints.

    Called first in `main`, because the transcript opens several lines later
    and argparse can write to stderr before either of them exists.
    """
    if sys.stdout is None:
        sys.stdout = _Null()
    if sys.stderr is None:
        sys.stderr = _Null()


def _make_output_lossy() -> None:
    """Never lose a finished reel to the console's codepage.

    The same guard step 0 puts on its board, for the same reason and with more
    at stake: a scouted title carrying an emoji reaches this module's summary
    through the topic, and a Windows console encoding cp1252 raises on it -
    after the encode. A mangled character in a printed line costs nothing; the
    log, the ledger and the run directory all keep the real text.
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(errors="replace")
        except (AttributeError, ValueError):
            pass


def log_path_for(when: datetime | None = None) -> Path:
    """Where this run's transcript goes: output/logs/<date>-<time>.log."""
    stamp = (when or datetime.now()).strftime("%Y-%m-%d-%H%M%S")
    return config.LOGS_DIR / f"{stamp}.log"


def prune_logs(keep_days: int | None = None, *, directory: Path | None = None) -> int:
    """Delete logs older than RUN_LOG_KEEP_DAYS. Returns how many went.

    Runs at the start of a run rather than the end, so a run killed by the
    scheduler's time limit still gets its housekeeping done the next day. Never
    raises: losing a log to a permission error is not a reason to lose a reel.
    """
    days = config.RUN_LOG_KEEP_DAYS if keep_days is None else keep_days
    if days <= 0:
        return 0
    cutoff = time.time() - days * 86400
    removed = 0
    try:
        entries = list((directory or config.LOGS_DIR).glob("*.log"))
    except OSError:
        return 0
    for entry in entries:
        try:
            if entry.stat().st_mtime < cutoff:
                entry.unlink()
                removed += 1
        except OSError:
            continue
    return removed


class Transcript:
    """Mirror everything printed inside the block into a log file.

    A no-op when `path` is None, so an interactive run can opt out and a test
    never has to care.
    """

    def __init__(self, path: Path | str | None):
        self.path = Path(path) if path else None
        self._handle = None
        self._saved = None

    def __enter__(self):
        if self.path is None:
            return self
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = open(self.path, "a", encoding="utf-8", errors="replace")
        self._saved = (sys.stdout, sys.stderr)
        sys.stdout = _Tee(self._saved[0], self._handle)
        sys.stderr = _Tee(self._saved[1], self._handle)
        return self

    def __exit__(self, *exc):
        if self._saved is not None:
            sys.stdout, sys.stderr = self._saved
            self._saved = None
        if self._handle is not None:
            self._handle.close()
            self._handle = None
        return False


# ---------------------------------------------------------------------------
# The ledger - one line per run, and what the once-per-day guard reads
# ---------------------------------------------------------------------------

def load_ledger(path: Path | str | None = None) -> dict:
    """The run ledger, or an empty one. Never raises."""
    data = visuals._read_json(
        Path(path or config.RUN_LEDGER_FILE),
        {"version": RUN_VERSION, "runs": []},
    )
    if not isinstance(data.get("runs"), list):
        data["runs"] = []
    return data


def append_run(record: dict, path: Path | str | None = None) -> list[str]:
    """Add one run to the ledger, oldest dropped past RUN_LEDGER_KEEP.

    Never fatal, and returns warnings rather than raising: the reel is on disk
    by the time this is called, and failing to write a note about it is not a
    reason to report the run as failed.
    """
    target = Path(path or config.RUN_LEDGER_FILE)
    ledger = load_ledger(target)
    ledger["version"] = RUN_VERSION
    ledger["runs"].append(ledger_entry(record))
    keep = config.RUN_LEDGER_KEEP
    if keep > 0:
        ledger["runs"] = ledger["runs"][-keep:]
    try:
        visuals._write_json(target, ledger)
    except OSError as exc:
        return [f"could not write the run ledger at {target}: {exc}"]
    return []


def ledger_entry(record: dict) -> dict:
    """The part of a run worth keeping in the index.

    Not the whole record - the run directory holds the manifests. This is the
    column you read down to see which of the last thirty days made nothing, and
    why, without opening thirty logs.
    """
    return {
        "day": record.get("day"),
        "started_at": record.get("started_at"),
        "seconds": record.get("seconds"),
        "status": record.get("status"),
        "reason": record.get("reason") or "",
        "topic": record.get("topic") or "",
        "run_id": record.get("run_id") or "",
        "video_path": record.get("video_path") or "",
        "log_path": record.get("log_path") or "",
        "scheduled": bool(record.get("scheduled")),
        "tts_characters": (record.get("spend") or {}).get("tts_characters", 0),
        "gemini": (record.get("spend") or {}).get("gemini") or {},
        "attempts": len(record.get("attempts") or []),
    }


def finished_today(when: date | None = None, *, ledger: dict | None = None) -> dict | None:
    """The reel already made today, if there is one.

    Two places are asked, because they are wrong in different directions. The
    ledger is the record of intent and knows about runs that produced nothing;
    the output directory is ground truth and survives the ledger being deleted -
    which is exactly what installing this step looks like on a project that has
    been making reels by hand for a week.
    """
    day = (when or date.today()).strftime(config.VIDEO_RUN_DATE_FORMAT)

    for entry in reversed((ledger if ledger is not None else load_ledger()).get("runs", [])):
        if entry.get("day") == day and entry.get("status") in ("ok", "partial"):
            return dict(entry)

    try:
        made = sorted((config.OUTPUT_DIR / day).glob(f"*/{config.VIDEO_FILENAME}"))
    except OSError:
        made = []
    if made:
        return {"day": day, "status": "ok", "run_id": made[0].parent.name,
                "video_path": str(made[0].resolve()), "topic": "",
                "reason": "found in output/, not in the ledger"}
    return None


def register_reel(record: dict) -> list[str]:
    """Write the features row and register the reel with step 9.

    Called the moment a reel exists, whether or not step 6 wrote its caption: a
    reel with no post text is still a reel that was made, and if it gets posted
    by hand it still has numbers worth recording against it.

    Nothing here can fail a run. The reel is finished and on disk by this
    point, and losing a bookkeeping row is not a reason to report that the
    evening produced nothing.
    """
    if not record.get("run_id"):
        return []
    try:
        features = perform.features_for(record)
        warnings = list(perform.write_features(features))
        ledger = perform.PerformanceLedger()
        ledger.register(features)
        return warnings + ledger.save()
    except (OSError, ValueError) as exc:
        return [f"could not register the reel with step 9: {exc}"]


# ---------------------------------------------------------------------------
# Choosing the topic - the decision a human was making with `--take N`
# ---------------------------------------------------------------------------

def choose_candidates(board: dict, *, attempts: int | None = None,
                      min_score: float | None = None) -> tuple[list[dict], list[str]]:
    """The candidates this run will try, best first.

    Step 0 has already done the hard part: it gated the niche, probed the stock
    providers for footage and ranked what survived. Taking the top of that is
    not a shrug, it is using the ranking - and anything cleverer here would be a
    second, worse ranking laid over a good one that nobody could read back.

    The one judgement left is whether the top of the board is worth making at
    all, which is RUN_TOPIC_MIN_SCORE's job, and it is off by default: a board
    that cleared the gate and the probe is a board worth making from.
    """
    wanted = max(1, config.RUN_TOPIC_ATTEMPTS if attempts is None else attempts)
    floor = config.RUN_TOPIC_MIN_SCORE if min_score is None else min_score
    notes: list[str] = []

    candidates = [c for c in (board.get("candidates") or []) if c.get("topic")]
    if not candidates:
        return [], ["the board came back empty"]

    if floor > 0:
        strong = [c for c in candidates if float(c.get("score") or 0) >= floor]
        if not strong:
            best = max(float(c.get("score") or 0) for c in candidates)
            notes.append(
                f"no candidate reached RUN_TOPIC_MIN_SCORE ({floor:g}) - the "
                f"best of {len(candidates)} scored {best:g}"
            )
            return [], notes
        if len(strong) < len(candidates):
            notes.append(
                f"{len(candidates) - len(strong)} candidate(s) under the "
                f"{floor:g} score floor were not considered"
            )
        candidates = strong

    return candidates[:wanted], notes


# ---------------------------------------------------------------------------
# The run
# ---------------------------------------------------------------------------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _announce(index: int, name: str, detail: str = "") -> None:
    """One line per step, stamped, so a log reads back as a timeline."""
    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"{stamp}  [{index}/{len(STEPS)}] {name:<9} {detail}", flush=True)


def _collect(record: dict, step: str, warnings) -> None:
    """Fold a step's warnings into the run's, saying which step raised them."""
    for warning in warnings or ():
        record["warnings"].append(f"{step}: {warning}")


def _read_insights(record: dict) -> None:
    """Ask step 9 for the numbers on the reels already out there. Never raises.

    Runs before the once-per-day guard, and that is the point: this reading is
    about reels posted on earlier days, so a day that makes no reel is still a
    day one of them has to be asked about. Behind the guard, a reel posted the
    evening before a skipped day would fall out of its window unread.

    Nothing here can cost a reel. The numbers are bookkeeping and the reel is
    the job, so a provider that is down, rate-limited or misconfigured leaves a
    warning on the run and the pipeline carries on.
    """
    if not config.RUN_FETCH_INSIGHTS:
        return
    # No provider is a choice, not a fault - the numbers can be typed in by
    # hand - so it is not worth a warning on every run for the rest of time.
    try:
        if perform.load_provider() is None:
            return
    except Exception:
        return

    stamp = datetime.now().strftime("%H:%M:%S")
    print(f"{stamp}  [-/{len(STEPS)}] insights  reading what is already out",
          flush=True)
    try:
        recorded, warnings = perform.fetch()
    except Exception as exc:  # a provider is third-party code
        record["warnings"].append(f"insights: {_one_line(exc)}")
        return
    _collect(record, "insights", warnings)
    print(f"          {recorded} reel(s) re-read", flush=True)

    # The same tail the read task runs, so a reel made by hand leaves the
    # report describing the ledger as it now stands rather than as it was.
    # Never fatal: the numbers are already saved, and this is the copy.
    try:
        _collect(record, "insights", perform.refresh_outputs())
    except Exception as exc:  # noqa: BLE001
        record["warnings"].append(f"insights: {_one_line(exc)}")


def _one_line(text) -> str:
    """An error boiled down to something a ledger column can hold.

    ffmpeg fails with several hundred lines of encoder statistics attached to
    the exception, and a `reason` field carrying all of it makes both the
    summary and runs.json unreadable - which matters most in exactly the case
    it happens, where the ledger is how you find out the run failed at all. The
    log already has the whole thing, verbatim; this is the sentence that says
    which step died and roughly why.
    """
    lines = [line.strip() for line in str(text).splitlines() if line.strip()]
    first = lines[0] if lines else ""
    if len(first) > _REASON_MAX_CHARS:
        first = first[:_REASON_MAX_CHARS].rstrip() + "..."
    if len(lines) > 1:
        first += f"  ({len(lines) - 1} more line(s) in the log)"
    return first


def _step_of(exc: BaseException) -> str:
    """Which step an error belongs to, for the record and the summary."""
    for step, kinds in (
        ("topics", (topics.TopicsError,)),
        ("script", (script_gen.ScriptGenerationError,)),
        ("tts", (tts.TTSError,)),
        ("visuals", (visuals.VisualsError,)),
        ("assemble", (assemble.AssemblyError,)),
        ("caption", (caption.CaptionError,)),
    ):
        if isinstance(exc, kinds):
            return step
    return "run"


def new_record(*, scheduled: bool = False, dry_run: bool = False,
               log_path: Path | str | None = None) -> dict:
    """A blank run record. Every key it will ever have is present here.

    Written out in full rather than grown key by key so that a run that dies in
    step 3 and one that finishes have the same shape, and anything reading the
    record can index it without guessing which half of the run happened.
    """
    return {
        "run_version": RUN_VERSION,
        "day": date.today().strftime(config.VIDEO_RUN_DATE_FORMAT),
        "started_at": _now(),
        "finished_at": None,
        "seconds": 0.0,
        "status": "failed",
        "reason": "",
        "scheduled": bool(scheduled),
        "dry_run": bool(dry_run),
        "topic": "",
        "topic_key": "",
        "topic_source": "",
        "topic_score": None,
        #: What step 0's scoring was made of, carried through for step 9. The
        #: names, not the arithmetic - see perform.features_for.
        "topic_components": {},
        #: The arm of the running trial this reel took, if one is running.
        "trial": {},
        "target_seconds": None,
        "run_id": "",
        "run_dir": "",
        "video_path": "",
        "script_path": "",
        "audio_path": "",
        "post_path": "",
        "log_path": str(log_path) if log_path else "",
        "duration_seconds": None,
        "attempts": [],
        "steps": {},
        "spend": {"tts_characters": 0, "downloaded_bytes": 0,
                  "gemini": {}},
        "warnings": [],
    }


def run_pipeline(topic: str | None = None, *, duration: int | None = None,
                 attempts: int | None = None, min_score: float | None = None,
                 once_per_day: bool | None = None, force: bool = False,
                 scheduled: bool = False, dry_run: bool = False,
                 no_phrases: bool = False,
                 log_path: Path | str | None = None,
                 ledger_path: Path | str | None = None) -> dict:
    """Take the pipeline from nothing to a finished reel.

    Args:
        topic: what to make. None scouts a board (step 0) and takes the best
            candidate on it, which is the unattended path.
        duration: target spoken seconds. None takes the length step 0 decided
            for the chosen topic, itself falling back to
            config.SCRIPT_DEFAULT_SECONDS.
        attempts: how many topics to try before giving up on the day. Defaults
            to config.RUN_TOPIC_ATTEMPTS, and is forced to 1 when `topic` was
            given - a topic you chose is not one this function may swap out.
        min_score: score floor for a candidate. Defaults to
            config.RUN_TOPIC_MIN_SCORE.
        once_per_day: refuse to start if today already has a reel. Defaults to
            config.RUN_ONCE_PER_DAY.
        force: run anyway. Overrides once_per_day, and nothing else.
        scheduled: mark the record as an unattended run. Changes no behaviour;
            it is how the ledger tells a 5pm run from one you started.
        dry_run: choose a topic, report the plan, and stop before step 2.
            Step 0 still runs exactly as it normally would, which is the point:
            a preview that decided the topic by a cheaper rule would preview a
            topic the real run is not going to make. It costs what step 0 costs
            and nothing else - no script, no narration, no encode.
        no_phrases: run step 0 with no Gemini calls at all, the same free mode
            as `topics.py --no-phrases`. Genuinely free, and it changes which
            topic comes out - the niche is then decided on shared words, which
            config.TOPIC_NICHE_MATCH explains is much stricter and wrong in
            both directions. Worth it to check plumbing; not worth it to
            decide what to make.
        log_path: recorded in the run record. Opening the log is the caller's
            job - see `Transcript` - so that importing this function does not
            hijack anybody's stdout.
        ledger_path: where the run is appended. Defaults to config.

    Returns:
        The run record: status, the topic, every path the run produced, what it
        spent, and every warning the six steps raised. A summary of it is also
        appended to config.RUN_LEDGER_FILE.

        `status` is one of:
          ok       a reel and its post text
          partial  a reel, but step 6 could not write the caption
          skipped  nothing was made and nothing was wrong - already made today,
                   a dry run, or no candidate cleared the score floor
          failed   a step raised

    Never raises for a failed run: the failure is the record's `status` and
    `reason`. The caller of an unattended run is a scheduler reading an exit
    code, and a traceback into a closed console helps nobody. Programming
    errors still propagate.
    """
    started = time.monotonic()
    record = new_record(scheduled=scheduled, dry_run=dry_run, log_path=log_path)
    # Zeroed here rather than at import, so a second run in the same process -
    # the tests, and `--attempts 2` after a refusal - does not inherit the
    # first one's tokens.
    script_gen.USAGE.reset()

    def finish(status: str, reason: str = "") -> dict:
        record["status"] = status
        record["reason"] = reason
        record["seconds"] = round(time.monotonic() - started, 1)
        record["finished_at"] = _now()
        # Read on every exit, not just the happy one: a run that refused three
        # scripts and gave up is exactly the run worth knowing the cost of.
        record["spend"]["gemini"] = script_gen.USAGE.as_dict()
        _collect(record, "ledger", append_run(record, ledger_path))
        return record

    prune_logs()

    # --- what the reels already out there did ------------------------------
    # Ahead of the guards on purpose: see `_read_insights`. A dry run is a
    # preview and makes no network calls of its own, so it makes none here.
    if not dry_run:
        _read_insights(record)

    # --- guards, before anything is spent ---------------------------------
    guard = config.RUN_ONCE_PER_DAY if once_per_day is None else once_per_day
    if guard and not force and not dry_run:
        made = finished_today(ledger=load_ledger(ledger_path))
        if made:
            known = made.get("topic") or made.get("run_id") or "?"
            print(f"today already has a reel: {known}")
            print("--force makes another one anyway.")
            return finish("skipped", "a reel was already made today")

    if not dry_run:
        try:
            config.require(*config.REQUIRED_RUN)
        except config.MissingConfigError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return finish("failed", _one_line(exc))

    # --- the one thing this reel does differently -------------------------
    # Chosen here rather than inside `_make_reel`, where it used to be, because
    # the `curiosity` arm has to be decided before the board is built: it
    # changes what step 0 charges, and so which topic wins. The later arms are
    # spent exactly where they were - `apply_arm` at step 2, `arm_assembly` at
    # step 5 - and now read a trial that was already settled rather than each
    # asking for one. Deciding it once also means a run that falls through to
    # a second candidate keeps the arm it started on, instead of re-deriving
    # it from a count that has not moved.
    trial = perform.choose_arm()
    if trial:
        record["trial"] = trial

    # --- step 0: what is this reel about? ---------------------------------
    step_started = time.monotonic()
    if topic:
        _announce(1, "topics", f"not scouted - topic given: {topic}")
        queue = [{"topic": topic.strip(), "key": "", "source": "given",
                  "score": None, "duration": duration}]
        notes: list[str] = []
    else:
        _announce(1, "topics",
                  "scouting" + (" (no Gemini calls)" if no_phrases else ""))
        try:
            if trial.get("name") in perform.BOARD_TRIALS:
                _announce(1, "topics",
                          f"trial \"{trial['name']}\": this board takes the "
                          f"\"{trial['arm']}\" arm")
            board = topics.find_topics(
                # Left on the configured modes even for a dry run. Turning
                # step 0's Gemini calls off makes the scout free, and it also
                # makes it pick a different topic - the niche then turns on
                # shared words, which puts a motorsport championship at the top
                # of an ocean-science board. A preview of a choice nobody will
                # make is worse than no preview.
                phrases="heuristic" if no_phrases else None,
                niche_match="terms" if no_phrases else None,
                # A board trial is the one arm that changes what gets made
                # rather than how it is made, so it is spent here, before a
                # candidate exists to apply anything to.
                **perform.arm_board(trial),
            )
        except (topics.TopicsError, config.MissingConfigError, ValueError) as exc:
            print(f"error: topics: {exc}", file=sys.stderr)
            return finish("failed", f"topics: {_one_line(exc)}")

        _collect(record, "topics", board.get("warnings"))
        _collect(record, "topics", topics.save_board(board))
        queue, notes = choose_candidates(board, attempts=attempts,
                                         min_score=min_score)
        for note in notes:
            record["warnings"].append(f"topics: {note}")
        if not queue:
            print("nothing on today's board is worth making.")
            for note in notes:
                print(f"  {note}")
            return finish("skipped", notes[0] if notes else "no candidates")
        _announce(1, "topics",
                  f"chose \"{queue[0]['topic']}\" from "
                  f"{board.get('stats', {}).get('shown', len(queue))} candidates")
    record["steps"]["topics"] = round(time.monotonic() - step_started, 1)

    if dry_run:
        best = queue[0]
        record.update(topic=best["topic"], topic_key=best.get("key") or "",
                      topic_source=best.get("source") or "",
                      topic_score=best.get("score"))
        _print_plan(queue, duration, no_phrases=no_phrases)
        return finish("skipped", "dry run - nothing was made")

    # --- steps 2-6, once per candidate ------------------------------------
    if topic:
        queue = queue[:1]

    last_error = ""
    for index, candidate in enumerate(queue, start=1):
        if index > 1:
            print(f"\nfalling through to candidate {index} of {len(queue)}: "
                  f"{candidate['topic']}")
        attempt = {"topic": candidate["topic"], "status": "failed",
                   "failed_at": "", "error": "", "refused_path": ""}
        record["attempts"].append(attempt)
        try:
            _make_reel(record, candidate, duration=duration, trial=trial)
        except _TOPIC_FAILURES as exc:
            step = _step_of(exc)
            attempt.update(failed_at=step, error=str(exc))
            last_error = f"{step}: {exc}"
            print(f"error: {last_error}", file=sys.stderr)

            # A draft the fact-check refused is evidence, and this is the last
            # moment it exists - the next candidate overwrites nothing but
            # nothing keeps it either. Its warnings come with it: the repair
            # rounds are recorded inside the claim report, which step 2 only
            # folds into the run when the script survives, so a refusal used
            # to be the one outcome whose reasoning never reached the log.
            _collect(record, step, (getattr(exc, "report", None) or {})
                     .get("warnings"))
            refused = getattr(exc, "script", "")
            if refused:
                try:
                    kept = script_gen.save_refused_script(
                        candidate["topic"], refused,
                        getattr(exc, "report", None) or {})
                except OSError as write_error:
                    _collect(record, step,
                             [f"the refused draft could not be kept "
                              f"({write_error})"])
                else:
                    attempt["refused_path"] = str(kept)
                    print(f"  the refused draft is kept in {kept}")

                # And the board is told, which it never used to be. A topic is
                # marked `used` at the first moment money has gone on it, and a
                # refused draft never reaches that line - so the whole cost of
                # a refusal was the -0.6 for having been on a board, and the
                # same topic came back to buy the same Pro call again. "What
                # the sound of the deep ocean actually is" did it five times.
                key = candidate.get("key") or ""
                if key:
                    _collect(record, "topics",
                             topics.refuse_topic(key, candidate["topic"]))
                    seen = topics.TopicLedger().refusals(key)
                    limit = config.TOPIC_MAX_REFUSALS
                    print(f"  the board will remember this refusal"
                          + (f" - {seen} of {limit}, and it is dropped at "
                             f"{limit}" if limit > 0 else ""))

            record["warnings"].append(
                f"{step}: gave up on \"{candidate['topic']}\" - {_one_line(exc)}"
            )
            continue
        except _RUN_FAILURES as exc:
            step = _step_of(exc)
            attempt.update(failed_at=step, error=str(exc))
            print(f"error: {step}: {exc}", file=sys.stderr)
            return finish("failed", f"{step}: {_one_line(exc)}")

        attempt["status"] = "ok"
        _collect(record, "perform", register_reel(record))
        made_post = bool(record["post_path"])
        return finish("ok" if made_post else "partial",
                      "" if made_post else "the reel was made, step 6 was not")

    return finish("failed", last_error or "no candidate could be made")


def _make_reel(record: dict, candidate: dict, *, duration: int | None,
               trial: dict | None = None) -> None:
    """Steps 2-6 for one topic. Fills `record` as it goes.

    Written to mutate the record rather than return one so that a run which
    dies in step 5 still reports the script and the narration it paid for.
    Those files exist and they were billed; a record that omitted them because
    the run failed later would be hiding money already spent.
    """
    topic_text = candidate["topic"]
    record.update(topic=topic_text, topic_key=candidate.get("key") or "",
                  topic_source=candidate.get("source") or "",
                  topic_score=candidate.get("score"),
                  topic_components=candidate.get("components") or {})

    target = duration if duration is not None else candidate.get("duration")

    # The one thing this reel does differently from the last one, so that step
    # 9 has something to compare. Chosen by `run_once` before the board was
    # built and handed down; this is where the arms that act on a *chosen*
    # topic are spent. Off by default; a trial only ever moves a choice the
    # pipeline was otherwise making the same way every day.
    trial = trial or {}
    if trial:
        target, hooks, shape, keep = perform.apply_arm(trial, duration=target)
        record["trial"] = trial
        # A board trial has already said its piece at step 1, where it was
        # actually spent. Saying it again here would claim step 2 did
        # something differently, and step 2 did not.
        if trial.get("name") not in perform.BOARD_TRIALS:
            _announce(2, "script", f"trial \"{trial['name']}\": this reel "
                                   f"takes the \"{trial['arm']}\" arm")
    else:
        hooks = shape = keep = None

    # --- step 2: the script -----------------------------------------------
    started = time.monotonic()
    record["target_seconds"] = target or config.SCRIPT_DEFAULT_SECONDS
    _announce(2, "script",
              f"{topic_text} ({target or config.SCRIPT_DEFAULT_SECONDS}s)")
    script = script_gen.save_script(
        script_gen.generate_script(topic_text, target,
                                   hooks=hooks, structure=shape,
                                   takeaway=keep)
    )
    record["script_path"] = script["script_path"]
    record["steps"]["script"] = round(time.monotonic() - started, 1)
    _announce(2, "script", f"{script['estimated_duration']}s estimated -> "
                           f"{Path(script['script_path']).name}")
    structure = script.get("structure") or {}
    if structure.get("revised"):
        _announce(2, "script", f"shape rewritten, {structure['score']:g} -> "
                               f"{structure['revised_score']:g}")

    # All four of step 2's passes report rather than raise, so their warnings
    # are the only trace of a hook that was kept, a rewrite that was thrown
    # away or a fact-check that could not be reached. They were being dropped.
    for pass_name in ("hook", "structure", "takeaway", "claims"):
        _collect(record, "script", (script.get(pass_name) or {}).get("warnings"))
    # Step 2's own warnings, as opposed to those of the four passes it runs:
    # what the script is after all of them have had their turn.
    _collect(record, "script", script.get("warnings"))

    claims = script.get("claims") or {}
    # A pass that did not run leaves no findings and no warnings, which reads
    # from the log exactly like a pass that ran and found nothing. They are
    # opposite facts about the reel: one script was checked and one was not,
    # and the deep-ocean reel shipped as the second while the log said neither.
    if claims.get("mode") != "llm":
        _collect(record, "script", [
            "the fact-check did not run (VERIFY_MODE="
            f"{claims.get('mode') or 'off'}) - nothing checked this script's "
            "claims before it was narrated"
        ])
    elif not claims.get("checked", True):
        # Configured on, asked, and never answered - a third state, and the
        # one that is easiest to miss, because the pass reports a mode of
        # "llm" and a rewrite it made. Said plainly here so it is not read off
        # the warning list as one more line about a timeout.
        _collect(record, "script", [
            "the fact-check could not be completed - this script was narrated "
            "with its claims unverified"
        ])
    if claims.get("revised"):
        _announce(2, "script", "a claim was corrected by the fact-check")
    for finding in claims.get("findings") or ():
        _announce(2, "script",
                  f"fact-check ({finding['kind']}): {finding['problem']}")

    # The topic is marked made here, at the first moment real money has gone on
    # it, rather than at the end of a run that worked. The case that decides it
    # is step 4 finding no footage: that topic would still be top of tomorrow's
    # board, and a daily job would buy the same script and the same narration
    # every evening to fail in the same place. Burning a topic that a broken
    # ffmpeg would have made is the cheaper mistake - that run produces no reel
    # either way, and it is the log you will be reading, not the board.
    if record["topic_key"]:
        _collect(record, "topics",
                 topics.take_topic(record["topic_key"], topic_text))

    # --- step 3: the narration --------------------------------------------
    started = time.monotonic()
    _announce(3, "tts", "rendering the narration")
    audio_path = config.AUDIO_DIR / f"{Path(script['script_path']).stem}.mp3"
    audio = tts.generate_audio(script["script"], str(audio_path))
    record["audio_path"] = audio["audio_path"]
    record["spend"]["tts_characters"] = audio["character_count"]
    record["steps"]["tts"] = round(time.monotonic() - started, 1)
    _announce(3, "tts", f"{audio['duration_seconds']}s of audio, "
                        f"{audio['character_count']} characters billed")

    # --- step 4: the footage ----------------------------------------------
    started = time.monotonic()
    _announce(4, "visuals", "sourcing footage")
    manifest = visuals.source_visuals(script, audio)
    _collect(record, "visuals", manifest.get("warnings"))
    stats = manifest.get("stats") or {}
    record["run_id"] = manifest.get("run_id") or ""
    record["spend"]["downloaded_bytes"] = stats.get("downloaded_bytes", 0)
    record["steps"]["visuals"] = round(time.monotonic() - started, 1)
    _announce(4, "visuals", f"{stats.get('assets', 0)} assets over "
                            f"{stats.get('beats', 0)} beats "
                            f"({stats.get('from_cache', 0)} from cache)")

    # --- step 5: the reel -------------------------------------------------
    started = time.monotonic()
    _announce(5, "assemble", "encoding")
    # The same trial, spent at the other end of the run: `apply_arm` answered
    # step 2 an hour ago, and a music or motion arm is only decided here.
    assembly = assemble.assemble_video(audio, manifest,
                                       **perform.arm_assembly(trial))
    _collect(record, "assemble", assembly.get("warnings"))
    record["run_id"] = assembly.get("run_id") or record["run_id"]
    record["video_path"] = assembly["video_path"]
    record["run_dir"] = str(Path(assembly["video_path"]).parent)
    record["duration_seconds"] = assembly.get("duration_seconds")
    record["steps"]["assemble"] = round(time.monotonic() - started, 1)
    _announce(5, "assemble", f"{assembly['duration_seconds']}s, "
                             f"{assembly['stats']['shots']} shots -> "
                             f"{Path(assembly['video_path']).name}")

    # --- step 6: the post text --------------------------------------------
    # The only step allowed to fail without failing the run. Everything
    # expensive is already on disk and correct; the caption is words, it is
    # regenerable for the price of one Gemini call, and throwing away a
    # finished reel because its hashtags did not come back would be absurd.
    started = time.monotonic()
    _announce(6, "caption", "writing the post text")
    try:
        post = caption.generate_caption(assembly)
    except (caption.CaptionError, config.MissingConfigError, OSError) as exc:
        record["warnings"].append(f"caption: {exc}")
        record["steps"]["caption"] = round(time.monotonic() - started, 1)
        _announce(6, "caption", f"failed - {exc}")
        print("the reel itself is finished; rerun step 6 alone with:\n"
              f"  python pipeline/caption.py -r \"{record['run_dir']}\"",
              file=sys.stderr)
        return
    _collect(record, "caption", post.get("warnings"))
    record["post_path"] = post.get("post_path") or str(
        Path(record["run_dir"]) / "post.txt"
    )
    record["steps"]["caption"] = round(time.monotonic() - started, 1)
    _announce(6, "caption", f"{post['chars']['caption']} characters, "
                            f"{len(post['hashtags'])} tags")


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def _print_plan(queue: list[dict], duration: int | None, *,
                no_phrases: bool = False) -> None:
    """What a real run would have made, and what it would have cost."""
    print(RULE)
    print("DRY RUN - the board was scouted, nothing was made")
    for index, candidate in enumerate(queue, start=1):
        marker = "->" if index == 1 else "  "
        seconds = duration or candidate.get("duration") or config.SCRIPT_DEFAULT_SECONDS
        score = candidate.get("score")
        print(f" {marker} {candidate['topic']}")
        print(f"      {seconds}s"
              + (f", score {score:g}" if isinstance(score, (int, float)) else "")
              + f", from {candidate.get('source') or 'given'}")
    if len(queue) > 1:
        print(f"    (the {len(queue) - 1} below it are the fall-through, used "
              f"only if step 4 finds no footage for the first)")
    print(RULE)
    if no_phrases:
        print("--no-phrases made step 0 free, and also made it choose on")
        print("shared words rather than meaning. A real run reads the niche")
        print("with Gemini and will very likely pick something else.")
        print(RULE)
    print("a real run would go on to generate a script, narrate it, source")
    print("footage, encode a reel and write the post text - billing Gemini")
    print("and ElevenLabs for the first two.")
    print(RULE)


def _print_summary(record: dict) -> None:
    """The whole run on one screen. The last thing written to the log."""
    print(RULE)
    print(f"STATUS  : {record['status'].upper()}"
          + (f" - {record['reason']}" if record["reason"] else ""))
    if record["topic"]:
        print(f"TOPIC   : {record['topic']}"
              + (f"  (from {record['topic_source']})"
                 if record["topic_source"] else ""))
    if record["video_path"]:
        print(f"REEL    : {record['video_path']}")
        print(f"LENGTH  : {record['duration_seconds']}s")
    if record["post_path"]:
        print(f"POST    : {record['post_path']}")
    if record["steps"]:
        print("STEPS   : " + ", ".join(f"{name} {record['steps'][name]:g}s"
                                       for name in STEPS
                                       if name in record["steps"]))
    print(f"TOOK    : {record['seconds']:g}s total")
    if record["spend"]["tts_characters"]:
        print(f"SPENT   : {record['spend']['tts_characters']} ElevenLabs "
              f"characters, {record['spend']['downloaded_bytes'] / 1024 / 1024:.1f} MB "
              f"of footage")
    gemini = record["spend"].get("gemini") or {}
    if gemini.get("calls"):
        # Thinking is billed at the output rate and is most of the output, so
        # it is named rather than folded silently into the total.
        print(f"TOKENS  : {gemini['calls']} Gemini call(s), "
              f"{gemini['input_tokens']:,} in / {gemini['output_tokens']:,} out"
              + (f" (of which {gemini['thinking_tokens']:,} thinking)"
                 if gemini.get("thinking_tokens") else ""))
    if len(record["attempts"]) > 1:
        print(f"TOPICS  : {len(record['attempts'])} tried - "
              + "; ".join(f"{a['topic']} ({a['status']})"
                          for a in record["attempts"]))
    if record.get("trial"):
        print(f"TRIAL   : {record['trial']['name']} = "
              f"{record['trial']['arm']}  (PERFORMANCE_EXPLORE)")
    if record["log_path"]:
        print(f"LOG     : {record['log_path']}")
    if record["run_id"] and not record["dry_run"]:
        # The reel is finished and nobody is watching; the useful thing to
        # leave behind is the exact line that closes the loop tomorrow.
        short = record["run_id"].rsplit("-", 1)[-1]
        print(f"RECORD  : python pipeline/perform.py {short} "
              f"--views N --watched N")
        # One word, from a fixed list. "not good enough to post" was written
        # into fourteen rows here and answered nothing; a code names the step
        # to go and change.
        print(f"          python pipeline/perform.py {short} --skipped "
              + "|".join(list(perform.SKIP_REASONS)[:6]))
    if record["warnings"]:
        print(RULE)
        print("WARNINGS")
        for warning in record["warnings"]:
            print(f"  - {warning}")
    print(RULE)


def _print_status(limit: int, ledger_path: Path | str | None = None) -> None:
    """The last `limit` runs, newest last. `--status`."""
    runs = load_ledger(ledger_path).get("runs", [])[-limit:]
    if not runs:
        print("no runs recorded yet.")
        print("  python pipeline/run.py            makes one now")
        print("  python pipeline/schedule.py install   makes one every day")
        return
    print(f"{'day':<12} {'status':<8} {'took':>7}  topic")
    print(RULE)
    for entry in runs:
        took = entry.get("seconds")
        mark = "*" if entry.get("scheduled") else " "
        print(f"{entry.get('day', '?'):<12} {entry.get('status', '?'):<8} "
              f"{(f'{took:g}s' if took else '-'):>7}{mark} "
              + (entry.get("topic") or entry.get("reason") or "")[:44])
    print(RULE)
    print("* ran from the scheduler")
    made = sum(1 for e in runs if e.get("status") in ("ok", "partial"))
    spent = sum(int(e.get("tts_characters") or 0) for e in runs)
    print(f"{made} of the last {len(runs)} runs made a reel; "
          f"{spent} ElevenLabs characters billed across them.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    _attach_console()
    _make_output_lossy()
    parser = argparse.ArgumentParser(
        description="Run the whole pipeline: topic to finished reel (step 8).",
    )
    parser.add_argument("topic", nargs="?", default=None,
                        help="what to make. Omitted, step 0 scouts a board "
                             "and the best candidate on it is made")
    parser.add_argument("-d", "--duration", type=int, default=None,
                        help="target spoken seconds (default: what step 0 "
                             "decided for the topic)")
    parser.add_argument("--attempts", type=int, default=None,
                        help=f"topics to try before giving up on the day "
                             f"(default: {config.RUN_TOPIC_ATTEMPTS})")
    parser.add_argument("--min-score", type=float, default=None,
                        help="score a candidate must reach to be made "
                             f"(default: {config.RUN_TOPIC_MIN_SCORE:g})")
    parser.add_argument("-f", "--force", action="store_true",
                        help="make a reel even if today already has one")
    parser.add_argument("--dry-run", action="store_true",
                        help="scout a board, choose a topic and print the "
                             "plan, without making anything")
    parser.add_argument("--no-phrases", action="store_true",
                        help="run step 0 with no Gemini calls at all (as "
                             "topics.py --no-phrases). Free, and it changes "
                             "which topic wins")
    parser.add_argument("--scheduled", action="store_true",
                        help="mark this as an unattended run (what the "
                             "scheduled task passes)")
    parser.add_argument("--no-log", action="store_true",
                        help="do not write output/logs/<stamp>.log")
    parser.add_argument("--status", action="store_true",
                        help="show how the last runs went, then exit")
    parser.add_argument("--limit", type=int, default=14, metavar="N",
                        help="runs to show with --status (default: 14)")
    parser.add_argument("--json", action="store_true",
                        help="print the run record as JSON instead of a summary")
    args = parser.parse_args(argv)

    if args.status:
        _print_status(max(1, args.limit))
        return 0

    # A dry run is a check you run at a console, and it makes nothing worth
    # keeping a transcript of.
    log_path = None if (args.no_log or args.dry_run) else log_path_for()

    with Transcript(log_path):
        # `--json` has to be pipeable, and the six steps print freely. So for
        # the length of the run everything human-readable is diagnostics and
        # goes to stderr, leaving stdout for the record alone. The log is
        # unaffected - the transcript is sitting under both streams.
        console = sys.stdout
        if args.json:
            sys.stdout = sys.stderr
        try:
            if log_path is not None:
                print(f"\n{RULE}")
                print(f"RUN     : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
                      + ("  (scheduled)" if args.scheduled else ""))
                print(RULE)
            record = run_pipeline(
                args.topic,
                duration=args.duration,
                attempts=args.attempts,
                min_score=args.min_score,
                force=args.force,
                scheduled=args.scheduled,
                dry_run=args.dry_run,
                no_phrases=args.no_phrases,
                log_path=log_path,
            )
            if not args.json and not args.dry_run:
                _print_summary(record)
        finally:
            sys.stdout = console
        if args.json:
            print(json.dumps(record, indent=2, ensure_ascii=False))

    # `skipped` is a success: nothing was made and nothing was wrong. Reporting
    # it as a failure would have Task Scheduler retry the run twice more, which
    # is precisely the doubling-up the once-per-day guard exists to prevent.
    return 0 if record["status"] in ("ok", "partial", "skipped") else 1


if __name__ == "__main__":
    raise SystemExit(main())
