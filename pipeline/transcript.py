"""Console + log capture for the two jobs that run with nobody watching.

Step 8 (`run.py`) and step 9's daily read (`perform.py --fetch`) are both
started by Task Scheduler under pythonw.exe, which hands a process no console
at all. Whatever they print is gone unless something sits under their stdout
and copies it to a file, and that is this module.

It used to live in `run.py`, which was fine while step 8 was the only job. The
read needed it too, and on 2026-10-01 it failed at 12:19 with exit 1 and left
nothing behind to say why. It could not just import `run.py`. That module
imports `perform.py` itself, which makes a cycle, and four more reel-making
steps (topics, tts, assemble, caption) that the read otherwise never touches.
Any one of them failing to import would have cost the day's reading. The
pieces moved here instead. `run.py` still re-exports every name, so nothing
that reached them through it has changed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402


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


def prune_logs(keep_days: int | None = None, *, directory: Path | None = None) -> int:
    """Delete logs older than RUN_LOG_KEEP_DAYS. Returns how many went.

    Runs at the start of a job rather than the end, so a job killed by the
    scheduler's time limit still gets its housekeeping done the next day.
    Never raises: losing a log to a permission error is not a reason to lose a
    reel, or a reading.
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
