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
              windowless: bool = True, now: datetime | None = None) -> str:
    """The Task Scheduler XML for the daily run.

    Every default comes from config, so the task that gets installed is the one
    `python config.py` says will be installed.
    """
    hour, minute = parse_time(time_of_day or config.SCHEDULE_TIME)
    start = next_occurrence(hour, minute, now=now)

    catch_up = config.SCHEDULE_CATCH_UP if catch_up is None else catch_up
    wake = config.SCHEDULE_WAKE if wake is None else wake
    minutes = (config.SCHEDULE_RETRY_MINUTES if retry_minutes is None
               else retry_minutes)
    count = (config.SCHEDULE_RETRY_COUNT if retry_count is None
             else retry_count)
    limit = (config.SCHEDULE_TIME_LIMIT_MINUTES if time_limit_minutes is None
             else time_limit_minutes)

    interpreter = python or python_command(windowless=windowless)
    arguments = f'"{RUNNER}" --scheduled'

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
    <Description>Reel Creator Agent - generate one reel a day (step 8). Output lands in output/&lt;date&gt;/ and is not posted anywhere.</Description>
    <URI>\\{escape(config.SCHEDULE_TASK_NAME)}</URI>
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
                 python: str | None = None) -> str:
    """The same job as a crontab entry, for when this leaves the laptop.

    Not installed by anything here - a crontab is one line and installing it
    behind someone's back on a machine this code has never seen is not a
    favour. Printed so it can be pasted.
    """
    hour, minute = parse_time(time_of_day or config.SCHEDULE_TIME)
    interpreter = python or sys.executable
    return (f"{minute} {hour} * * * cd '{config.BASE_DIR}' && "
            f"'{interpreter}' pipeline/run.py --scheduled")


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


def install(*, task_name: str | None = None, **kwargs) -> str:
    """Write the task definition and register it. Returns schtasks' reply.

    Replaces an existing task of the same name, which is what makes `install`
    the way to *change* the time as well as set it: there is one task, and
    running install again is how it gets edited.
    """
    name = task_name or config.SCHEDULE_TASK_NAME
    xml = build_xml(**kwargs)

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

def _print_installed(name: str, xml_time: str, interpreter: Path) -> None:
    hour, minute = parse_time(xml_time)
    first = next_occurrence(hour, minute)
    print(RULE)
    print(f"INSTALLED : {name}")
    print(f"RUNS      : every day at {xml_time}")
    print(f"FIRST     : {first.strftime('%a %d %b, %H:%M')}")
    print(f"COMMAND   : {interpreter} \"{RUNNER}\" --scheduled")
    print(f"OUTPUT    : {config.OUTPUT_DIR}\\<date>\\<run id>\\")
    print(f"LOGS      : {config.LOGS_DIR}")
    print(RULE)
    print("It generates the reel and stops there - nothing is posted.")
    print("Check on it with:")
    print("  python pipeline/schedule.py status")
    print("  python pipeline/run.py --status")
    print("Try the whole thing once, right now, without waiting:")
    print("  python pipeline/schedule.py run-now")


def _print_status(name: str) -> None:
    fields = query(name)
    print(RULE)
    if fields is None:
        print(f"TASK    : {name} is not installed")
        print(RULE)
        print("  python pipeline/schedule.py install")
        return

    result = fields.get("last result", "")
    print(f"TASK    : {name}")
    print(f"STATE   : {fields.get('scheduled task state') or fields.get('status', '?')}")
    print(f"NEXT    : {fields.get('next run time', '?')}")
    print(f"LAST    : {fields.get('last run time', '?')}")
    print(f"RESULT  : {result}"
          + (f"  ({_LAST_RESULT[result]})" if result in _LAST_RESULT else ""))
    print(f"RUNS    : {fields.get('task to run', '?')}")
    print(RULE)

    # The OS knows whether the process started. Only the ledger knows whether a
    # reel came out of it, which is the question actually being asked.
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
        description="Install the daily reel run into the OS scheduler.",
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
                        help=f"task name (default: {config.SCHEDULE_TASK_NAME})")
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

    name = args.name or config.SCHEDULE_TASK_NAME
    moment = args.time or config.SCHEDULE_TIME
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
                print(build_xml(**options))
            else:
                print("# not Windows - the equivalent crontab line is:")
                print(crontab_line(time_of_day=moment,
                                   python=args.python))
            return 0

        if args.action == "status":
            _print_status(name)
            return 0

        if args.action == "uninstall":
            uninstall(name)
            print(f"removed {name}. Nothing will run by itself any more.")
            return 0

        if args.action == "run-now":
            run_now(name)
            print(f"started {name}.")
            print("It runs in the background - watch it with:")
            print(f"  python pipeline/run.py --status")
            print(f"  the newest file in {config.LOGS_DIR}")
            return 0

        install(task_name=name, **options)
        _print_installed(name, moment,
                         options["python"]
                         or python_command(windowless=not args.console))
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
