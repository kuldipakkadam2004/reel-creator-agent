"""Register step 8 with the machine's own scheduler, so the reel happens daily.

This module schedules nothing by itself and runs nothing by itself. It writes a
task definition, hands it to the OS, and gets out of the way - which is the
whole design decision. A long-lived Python process that sleeps until 5pm would
work until the first reboot, and then it would need something to start it at
boot, and the thing that starts programs at boot on this machine is Task
Scheduler. So it is Task Scheduler, directly, with no daemon of ours in the
middle.

It goes in via XML rather than `schtasks /Create /SC DAILY`, because the flags
that matter for a *local* daily job are not on the command line:

  StartWhenAvailable   run as soon as the machine is back, if 5pm was missed.
                       A desktop is asleep or off at 5pm more often than not,
                       and without this those evenings silently make nothing.
  MultipleInstancesPolicy=IgnoreNew
                       a catch-up run cannot land on top of a live one.
  RestartOnFailure     a provider having a bad ten minutes, or a laptop that
                       has not found the wifi yet, is worth one more go.
  ExecutionTimeLimit   a hung run cannot hold the day open forever.

Standalone use:
    python pipeline/schedule.py install          # daily at SCHEDULE_TIME
    python pipeline/schedule.py install --time 18:30
    python pipeline/schedule.py status
    python pipeline/schedule.py run-now          # fire it once, now
    python pipeline/schedule.py preview          # print the XML, install nothing
    python pipeline/schedule.py uninstall
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from xml.sax.saxutils import escape

# Allow both `python pipeline/schedule.py` and `python -m pipeline.schedule`
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from pipeline import perform  # noqa: E402
from pipeline import run as run_step  # noqa: E402

RULE = "-" * 68

#: Whether the OS scheduler this module drives is the one that exists here.
#: A module-level constant rather than an `os.name` check at each call site,
#: because the tests have to be able to stand somewhere else for a moment and
#: reassigning `os.name` itself takes pathlib down with it.
WINDOWS = os.name == "nt"

#: The task calls this, with `--scheduled` so the ledger can tell a 5pm run
#: from one you started at a console.
RUNNER = config.BASE_DIR / "pipeline" / "run.py"

#: The other job that can go on a clock: step 9's read. It makes nothing, spends
#: nothing on Gemini or ElevenLabs, and writes only the ledger - so it is the
#: one piece of this pipeline that is safe to automate without also deciding
#: that a video gets published somewhere.
READER = config.BASE_DIR / "pipeline" / "perform.py"

#: The jobs `install` knows how to register. Two, not one, because they are
#: separate decisions with opposite consequences for getting them wrong - see
#: `job_spec`.
JOBS = ("reel", "read")

#: Task Scheduler's own schema. `1.2` is the version that has every setting
#: used here and is understood by Windows 7 onwards - there is nothing to gain
#: from a later one.
TASK_VERSION = "1.2"

#: Windows exit codes worth translating in `status`. Anything else is printed
#: raw rather than guessed at.
_LAST_RESULT = {
    "0": "the run succeeded",
    "1": "the run failed - read the log it names in output/logs/",
    "267009": "currently running",
    "267010": "not run yet (the task is disabled)",
    "267011": "not run yet",
    "267014": "the last run was stopped by the user",
    "2147942401": "the command could not be found - reinstall the task",
    "2147943645": "the machine was asleep or logged out at the trigger time",
}


class ScheduleError(RuntimeError):
    """The task could not be installed, removed or read."""


def job_spec(job: str = "reel") -> dict:
    """Everything that differs between the reel task and the read task.

    Read out of `config` on every call rather than baked in at import, so that
    what gets installed is what `python config.py` currently says - the same
    contract `build_xml` already had for the other settings.

    **Why these are two tasks and not one with a flag.** Making a reel is a
    content decision, and somebody may reasonably want to keep it by hand.
    Reading how the last reel did is not a decision at all, and the cost of
    missing it is permanent: a reading has to land between
    PERFORMANCE_MIN_AGE_HOURS and PERFORMANCE_WINDOW_HOURS or that reel can
    never be compared with any other, ever. Measured 2026-09-16 with the read
    riding on step 8 and step 8 run by hand: 0 of 12 posted reels were
    comparable, 7 of them read only past 48h. So the read is worth a clock
    even where the reel is not, and each has to be installable and removable
    without touching the other.
    """
    if job not in JOBS:
        raise ScheduleError(
            f"unknown job {job!r} - it is one of {', '.join(JOBS)}"
        )
    if job == "read":
        return {
            "job": "read",
            "runner": READER,
            "arguments": f'"{READER}" --fetch',
            "task_name": config.SCHEDULE_READ_TASK_NAME,
            "time_of_day": config.SCHEDULE_READ_TIME,
            "time_limit": config.SCHEDULE_READ_TIME_LIMIT_MINUTES,
            "description": (
                "Reel Creator Agent - read how already-posted reels did "
                "(step 9) and write the numbers to the ledger. It makes no "
                "video, posts nothing, and spends nothing on Gemini or "
                "ElevenLabs."
            ),
            "makes": "It reads numbers into the ledger and makes nothing.",
        }
    return {
        "job": "reel",
        "runner": RUNNER,
        "arguments": f'"{RUNNER}" --scheduled',
        "task_name": config.SCHEDULE_TASK_NAME,
        "time_of_day": config.SCHEDULE_TIME,
        "time_limit": config.SCHEDULE_TIME_LIMIT_MINUTES,
        "description": (
            "Reel Creator Agent - generate one reel a day (step 8). Output "
            "lands in output/<date>/ and is not posted anywhere."
        ),
        "makes": "It generates the reel and stops there - nothing is posted.",
    }


# ---------------------------------------------------------------------------
# The task definition
# ---------------------------------------------------------------------------

def parse_time(value: str) -> tuple[int, int]:
    """"17:00" -> (17, 0). 24-hour, because a scheduler has no opinions."""
    try:
        moment = datetime.strptime(value.strip(), "%H:%M")
    except (ValueError, AttributeError) as exc:
        raise ScheduleError(
            f"--time must be 24-hour HH:MM, got {value!r} (5pm is 17:00)"
        ) from exc
    return moment.hour, moment.minute


def next_occurrence(hour: int, minute: int, *, now: datetime | None = None) -> datetime:
    """The next time that clock reading comes round.

    The trigger deliberately starts in the future rather than at midnight
    today. StartWhenAvailable treats a start boundary already in the past as a
    missed run and can fire a catch-up immediately - so installing the task at
    lunchtime would make a reel at lunchtime, which is a surprising way to
    spend ElevenLabs credits five seconds after typing `install`.
    """
    now = now or datetime.now()
    today = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return today if today > now else today + timedelta(days=1)


def python_command(*, windowless: bool = True) -> Path:
    """The interpreter the task should call.

    `sys.executable` and not "python": PATH belongs to whichever shell is open
    at the time, and the task runs in no shell at all. If this was installed
    from inside a virtualenv, this is that virtualenv's interpreter, which is
    the one that has the project's dependencies.

    pythonw.exe when it is there, so a run at 5pm does not throw a console
    window over whatever is on screen. Nothing is lost by it: step 8 writes its
    entire transcript to output/logs/ either way, and its _Tee is built to
    tolerate the absent stdout that pythonw.exe hands it.
    """
    executable = Path(sys.executable)
    if windowless:
        windowless_exe = executable.with_name("pythonw.exe")
        if windowless_exe.is_file():
            return windowless_exe
    return executable


def current_user() -> str:
    """DOMAIN\\user for the Principal. Falls back to the bare username."""
    domain = os.environ.get("USERDOMAIN", "").strip()
    user = os.environ.get("USERNAME", "").strip()
    if not user:
        import getpass
        user = getpass.getuser()
    return f"{domain}\\{user}" if domain else user


def build_xml(*, time_of_day: str | None = None, python: Path | None = None,
              catch_up: bool | None = None, wake: bool | None = None,
              retry_minutes: int | None = None, retry_count: int | None = None,
              time_limit_minutes: int | None = None,
              windowless: bool = True, now: datetime | None = None,
              job: str = "reel") -> str:
    """The Task Scheduler XML for one of the two daily jobs.

    Every default comes from config, so the task that gets installed is the one
    `python config.py` says will be installed. `job` picks which set of those
    defaults applies - see `job_spec`.
    """
    spec = job_spec(job)
    hour, minute = parse_time(time_of_day or spec["time_of_day"])
    start = next_occurrence(hour, minute, now=now)

    catch_up = config.SCHEDULE_CATCH_UP if catch_up is None else catch_up
    wake = config.SCHEDULE_WAKE if wake is None else wake
    minutes = (config.SCHEDULE_RETRY_MINUTES if retry_minutes is None
               else retry_minutes)
    count = (config.SCHEDULE_RETRY_COUNT if retry_count is None
             else retry_count)
    limit = (spec["time_limit"] if time_limit_minutes is None
             else time_limit_minutes)

    interpreter = python or python_command(windowless=windowless)
    arguments = spec["arguments"]

    # RestartOnFailure with a count of 0 is not "no retries", it is invalid.
    # Leaving the element out entirely is how you say no retries.
    restart = ""
    if count > 0 and minutes > 0:
        restart = (f"    <RestartOnFailure>\n"
                   f"      <Count>{count}</Count>\n"
                   f"      <Interval>PT{minutes}M</Interval>\n"
                   f"    </RestartOnFailure>\n")

    return f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="{TASK_VERSION}" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <RegistrationInfo>
    <Author>{escape(current_user())}</Author>
    <Description>{escape(spec["description"])}</Description>
    <URI>\\{escape(spec["task_name"])}</URI>
  </RegistrationInfo>
  <Principals>
    <Principal id="Author">
      <UserId>{escape(current_user())}</UserId>
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>LeastPrivilege</RunLevel>
    </Principal>
  </Principals>
  <Settings>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT{limit}M</ExecutionTimeLimit>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
{restart}    <StartWhenAvailable>{str(catch_up).lower()}</StartWhenAvailable>
    <RunOnlyIfNetworkAvailable>true</RunOnlyIfNetworkAvailable>
    <AllowHardTerminate>true</AllowHardTerminate>
    <AllowStartOnDemand>true</AllowStartOnDemand>
    <Enabled>true</Enabled>
    <Hidden>false</Hidden>
    <RunOnlyIfIdle>false</RunOnlyIfIdle>
    <WakeToRun>{str(wake).lower()}</WakeToRun>
    <IdleSettings>
      <StopOnIdleEnd>false</StopOnIdleEnd>
      <RestartOnIdle>false</RestartOnIdle>
    </IdleSettings>
    <Priority>7</Priority>
  </Settings>
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>{start.strftime('%Y-%m-%dT%H:%M:%S')}</StartBoundary>
      <Enabled>true</Enabled>
      <ScheduleByDay>
        <DaysInterval>1</DaysInterval>
      </ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Actions Context="Author">
    <Exec>
      <Command>{escape(str(interpreter))}</Command>
      <Arguments>{escape(arguments)}</Arguments>
      <WorkingDirectory>{escape(str(config.BASE_DIR))}</WorkingDirectory>
    </Exec>
  </Actions>
</Task>
"""


def crontab_line(*, time_of_day: str | None = None,
                 python: str | None = None, job: str = "reel") -> str:
    """The same job as a crontab entry, for when this leaves the laptop.

    Not installed by anything here - a crontab is one line and installing it
    behind someone's back on a machine this code has never seen is not a
    favour. Printed so it can be pasted.
    """
    spec = job_spec(job)
    hour, minute = parse_time(time_of_day or spec["time_of_day"])
    interpreter = python or sys.executable
    script = spec["runner"].name
    tail = "--fetch" if job == "read" else "--scheduled"
    return (f"{minute} {hour} * * * cd '{config.BASE_DIR}' && "
            f"'{interpreter}' pipeline/{script} {tail}")


# ---------------------------------------------------------------------------
# Talking to schtasks
# ---------------------------------------------------------------------------

def _schtasks(*args: str) -> tuple[int, str]:
    """Run schtasks and return (exit code, its output).

    Decoded leniently: schtasks writes in the console's OEM codepage, and a
    task name or a path with a character outside it is not a reason to lose the
    error message that came with it.
    """
    if not WINDOWS:
        raise ScheduleError(
            "schtasks is Windows-only. On this machine the equivalent is a "
            "crontab line - `python pipeline/schedule.py preview` prints it."
        )
    try:
        completed = subprocess.run(
            ["schtasks", *args], capture_output=True, check=False,
        )
    except FileNotFoundError as exc:
        raise ScheduleError(
            "schtasks.exe was not found on PATH, which should not happen on "
            "Windows. Install the task by hand from Task Scheduler instead - "
            "`python pipeline/schedule.py preview` prints the XML to import."
        ) from exc
    output = b"\n".join(p for p in (completed.stdout, completed.stderr) if p)
    return completed.returncode, output.decode(errors="replace").strip()


def install(*, task_name: str | None = None, job: str = "reel",
            **kwargs) -> str:
    """Write the task definition and register it. Returns schtasks' reply.

    Replaces an existing task of the same name, which is what makes `install`
    the way to *change* the time as well as set it: there is one task per job,
    and running install again is how it gets edited. `job` picks which of the
    two - they have different names, so installing one never disturbs the
    other.
    """
    spec = job_spec(job)
    name = task_name or spec["task_name"]
    xml = build_xml(job=job, **kwargs)

    # UTF-16 with a BOM, matching the declaration in the document. schtasks
    # rejects a UTF-8 file as malformed XML rather than saying so plainly, and
    # that error has cost people entire afternoons.
    handle = tempfile.NamedTemporaryFile(
        "w", suffix=".xml", encoding="utf-16", delete=False, newline="\r\n",
    )
    try:
        handle.write(xml)
        handle.close()
        code, output = _schtasks("/Create", "/TN", name, "/XML", handle.name, "/F")
    finally:
        try:
            os.unlink(handle.name)
        except OSError:
            pass

    if code != 0:
        raise ScheduleError(f"could not create the task {name!r}: {output}")
    return output


def uninstall(task_name: str | None = None) -> str:
    """Remove the task. Missing is not an error - it is the desired state."""
    name = task_name or config.SCHEDULE_TASK_NAME
    code, output = _schtasks("/Delete", "/TN", name, "/F")
    if code != 0 and "cannot find" not in output.lower():
        raise ScheduleError(f"could not delete the task {name!r}: {output}")
    return output


def query(task_name: str | None = None) -> dict | None:
    """What the OS knows about the task, or None if it is not installed."""
    name = task_name or config.SCHEDULE_TASK_NAME
    code, output = _schtasks("/Query", "/TN", name, "/FO", "LIST", "/V")
    if code != 0:
        return None

    fields: dict[str, str] = {}
    for line in output.splitlines():
        key, sep, value = line.partition(":")
        if sep and key.strip():
            fields.setdefault(key.strip().lower(), value.strip())
    return fields


def run_now(task_name: str | None = None) -> str:
    """Fire the task immediately, exactly as 5pm would."""
    name = task_name or config.SCHEDULE_TASK_NAME
    code, output = _schtasks("/Run", "/TN", name)
    if code != 0:
        raise ScheduleError(f"could not start the task {name!r}: {output}")
    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _print_installed(name: str, xml_time: str, interpreter: Path,
                     job: str = "reel") -> None:
    spec = job_spec(job)
    hour, minute = parse_time(xml_time)
    first = next_occurrence(hour, minute)
    flag = " --read" if job == "read" else ""
    print(RULE)
    print(f"INSTALLED : {name}")
    print(f"RUNS      : every day at {xml_time}")
    print(f"FIRST     : {first.strftime('%a %d %b, %H:%M')}")
    print(f"COMMAND   : {interpreter} {spec['arguments']}")
    if job == "read":
        print(f"WRITES    : {config.PERFORMANCE_FILE}")
    else:
        print(f"OUTPUT    : {config.OUTPUT_DIR}\\<date>\\<run id>\\")
        print(f"LOGS      : {config.LOGS_DIR}")
    print(RULE)
    print(spec["makes"])
    print("Check on it with:")
    print(f"  python pipeline/schedule.py status{flag}")
    print("  python pipeline/perform.py" if job == "read"
          else "  python pipeline/run.py --status")
    if job == "read":
        # The one fact that explains why this task is worth having even
        # where the reel task is not.
        print(f"A reading counts only between "
              f"{config.PERFORMANCE_MIN_AGE_HOURS}h and "
              f"{config.PERFORMANCE_WINDOW_HOURS}h after a reel is "
              f"posted, and a daily")
        print("trigger lands in that window exactly once - which is the "
              "whole reason to put this on a clock.")
    print("Try it once, right now, without waiting:")
    print(f"  python pipeline/schedule.py run-now{flag}")


def _print_read_state() -> None:
    """What the read task is for, in the only terms that matter: the ledger.

    The OS can say the process started. Only this can say whether any reel got
    a reading inside the window it had to be inside, which is the entire point
    of putting the read on a clock.
    """
    try:
        ledger = perform.PerformanceLedger()
    except Exception as exc:  # a ledger that will not load is not a crash here
        print(f"could not read the ledger: {exc}")
        return
    rows = ledger.rows()
    posted = [r for r in rows if r.get("status") == "posted"]
    lo = config.PERFORMANCE_MIN_AGE_HOURS
    hi = config.PERFORMANCE_WINDOW_HOURS
    inside = [r for r in posted
              if r.get("hours_after_post") is not None
              and lo <= float(r["hours_after_post"]) <= hi]
    waiting = ledger.unsettled()
    print(f"posted   : {len(posted)} reel(s), {len(inside)} of them read "
          f"inside {lo}-{hi}h and so comparable")
    print(f"waiting  : {len(waiting)} reel(s) this task would ask about tonight")
    print("  python pipeline/perform.py          which ones, and when each is due")
    print("  python pipeline/perform.py --report what they add up to")


def _print_status(name: str, job: str = "reel") -> None:
    flag = " --read" if job == "read" else ""
    fields = query(name)
    print(RULE)
    if fields is None:
        print(f"TASK    : {name} is not installed")
        print(RULE)
        print(f"  python pipeline/schedule.py install{flag}")
        return

    result = fields.get("last result", "")
    print(f"TASK    : {name}")
    print(f"STATE   : {fields.get('scheduled task state') or fields.get('status', '?')}")
    print(f"NEXT    : {fields.get('next run time', '?')}")
    print(f"LAST    : {fields.get('last run time', '?')}")
    # Exit 1 means different things to the two jobs, and pointing somebody at
    # output/logs/ after a failed *read* sends them to a directory the read
    # task never writes to. It leaves no transcript at all - under pythonw.exe
    # its stdout goes nowhere - so what it did is read out of the ledger below.
    meaning = dict(_LAST_RESULT)
    if job == "read":
        meaning["1"] = ("the read failed - run `python pipeline/perform.py "
                        "--fetch` by hand to see why")
    print(f"RESULT  : {result}"
          + (f"  ({meaning[result]})" if result in meaning else ""))
    print(f"RUNS    : {fields.get('task to run', '?')}")
    print(RULE)

    # The OS knows whether the process started. Only the ledger knows whether a
    # reel came out of it, which is the question actually being asked.
    if job == "read":
        _print_read_state()
        return

    runs = run_step.load_ledger().get("runs", [])
    if not runs:
        print("no runs recorded yet - output/runs.json is empty.")
    else:
        last = runs[-1]
        print(f"last run : {last.get('day')} {last.get('status')}"
              + (f" - {last.get('topic') or last.get('reason')}"
                 if (last.get("topic") or last.get("reason")) else ""))
        print("  python pipeline/run.py --status   for the rest")


def main(argv: list[str] | None = None) -> int:
    run_step._attach_console()
    run_step._make_output_lossy()
    parser = argparse.ArgumentParser(
        description="Install a daily job into the OS scheduler: the reel run "
                    "(step 8), or --read, which only reads how already-posted "
                    "reels did (step 9) and makes nothing.",
    )
    parser.add_argument("action",
                        choices=("install", "uninstall", "status", "run-now",
                                 "preview"),
                        help="install: register the daily task (and the way to "
                             "change it). preview: print the definition "
                             "without touching anything")
    parser.add_argument("-t", "--time", default=None, metavar="HH:MM",
                        help=f"24-hour local time to run "
                             f"(default: {config.SCHEDULE_TIME})")
    parser.add_argument("-n", "--name", default=None,
                        help=f"task name (default: {config.SCHEDULE_TASK_NAME}, "
                             f"or {config.SCHEDULE_READ_TASK_NAME} with --read)")
    parser.add_argument("--read", action="store_true",
                        help="act on the reading task instead of the reel "
                             "task: it asks the provider how already-posted "
                             "reels did and writes the ledger, generating no "
                             "video and spending nothing on Gemini or "
                             "ElevenLabs. Its own task name, so the two "
                             "install and uninstall independently")
    parser.add_argument("--python", default=None, metavar="PATH",
                        help="interpreter to run it with (default: the one "
                             "installing it, windowless if available)")
    parser.add_argument("--console", action="store_true",
                        help="run under python.exe, showing a console window")
    parser.add_argument("--wake", dest="wake", action="store_true", default=None,
                        help="wake the machine from sleep to run")
    parser.add_argument("--no-wake", dest="wake", action="store_false",
                        help="never wake the machine (the default)")
    parser.add_argument("--no-catch-up", dest="catch_up", action="store_false",
                        default=None,
                        help="skip the day entirely if the machine was off at "
                             "the scheduled time")
    args = parser.parse_args(argv)

    job = "read" if args.read else "reel"
    spec = job_spec(job)
    name = args.name or spec["task_name"]
    moment = args.time or spec["time_of_day"]
    options = dict(
        time_of_day=moment,
        python=Path(args.python) if args.python else None,
        catch_up=args.catch_up,
        wake=args.wake,
        windowless=not args.console,
    )

    try:
        if args.action == "preview":
            if WINDOWS:
                print(build_xml(job=job, **options))
            else:
                print("# not Windows - the equivalent crontab line is:")
                print(crontab_line(time_of_day=moment,
                                   python=args.python, job=job))
            return 0

        if args.action == "status":
            _print_status(name, job)
            return 0

        if args.action == "uninstall":
            uninstall(name)
            # Deliberately not "nothing runs by itself any more": the other
            # task may well still be installed, and saying otherwise about a
            # scheduler is the kind of wrong that is found out late.
            other = job_spec("reel" if job == "read" else "read")
            print(f"removed {name}.")
            print(f"  python pipeline/schedule.py status"
                  + (" --read" if job != "read" else "")
                  + f"   whether {other['task_name']} is still installed")
            return 0

        if args.action == "run-now":
            run_now(name)
            print(f"started {name}.")
            print("It runs in the background - watch it with:")
            if job == "read":
                print("  python pipeline/perform.py")
            else:
                print(f"  python pipeline/run.py --status")
                print(f"  the newest file in {config.LOGS_DIR}")
            return 0

        install(task_name=name, job=job, **options)
        _print_installed(name, moment,
                         options["python"]
                         or python_command(windowless=not args.console),
                         job)
        return 0

    except ScheduleError as exc:
        print(f"error: {exc}", file=sys.stderr)
        if not WINDOWS:
            print("\n# the crontab line for this machine:", file=sys.stderr)
            print(crontab_line(time_of_day=moment, python=args.python),
                  file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
