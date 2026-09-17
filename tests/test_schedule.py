"""Tests for pipeline/schedule.py.

Nothing here installs anything. `schtasks` is stubbed, so the suite never
registers a task, never deletes one and never starts a run - what is under test
is the definition that would be handed to Windows, and the handling of what it
says back.

The definition itself was checked against the real Task Scheduler once, by
creating it under a throwaway name and deleting it. If the XML is changed in a
way these tests do not cover, do that again:

    python -c "from pipeline import schedule; \
        print(schedule.install(task_name='SchemaCheck')); \
        print(schedule.uninstall('SchemaCheck'))"

    python -m pytest tests/test_schedule.py -v
    python tests/test_schedule.py     # works without pytest installed
"""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from xml.etree import ElementTree

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest  # noqa: E402

import config  # noqa: E402
from pipeline import schedule  # noqa: E402

NS = {"t": "http://schemas.microsoft.com/windows/2004/02/mit/task"}


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------

class FakeCompleted:
    def __init__(self, returncode=0, stdout=b"", stderr=b""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@pytest.fixture
def schtasks(monkeypatch):
    """Stub subprocess.run; record the argv and hand back a canned reply."""
    calls = []
    replies = []

    def fake_run(args, **kwargs):
        # Read the definition before install() deletes the temp file.
        payload = None
        if "/XML" in args:
            payload = Path(args[args.index("/XML") + 1]).read_text(
                encoding="utf-16")
        calls.append({"args": list(args), "xml": payload})
        return replies.pop(0) if replies else FakeCompleted(0, b"SUCCESS: done")

    monkeypatch.setattr(schedule, "WINDOWS", True)
    monkeypatch.setattr(schedule.subprocess, "run", fake_run)
    fake_run.calls = calls
    fake_run.replies = replies
    return fake_run


def _parse(xml: str) -> ElementTree.Element:
    return ElementTree.fromstring(xml)


def _text(tree: ElementTree.Element, path: str) -> str:
    found = tree.find(path, NS)
    return "" if found is None else (found.text or "")


# ---------------------------------------------------------------------------
# The time
# ---------------------------------------------------------------------------

def test_parse_time_reads_24_hour_clock():
    assert schedule.parse_time("17:00") == (17, 0)
    assert schedule.parse_time("06:30") == (6, 30)


@pytest.mark.parametrize("bad", ["5pm", "17", "25:00", "17:61", "", None])
def test_parse_time_rejects_anything_else(bad):
    with pytest.raises(schedule.ScheduleError) as excinfo:
        schedule.parse_time(bad)
    assert "HH:MM" in str(excinfo.value)


def test_next_occurrence_is_today_when_the_time_is_still_coming():
    now = datetime(2026, 8, 24, 9, 0)
    assert schedule.next_occurrence(17, 0, now=now) == datetime(2026, 8, 24, 17, 0)


def test_next_occurrence_rolls_to_tomorrow_once_the_time_has_passed():
    # Installing at 6pm must not leave a start boundary in the past:
    # StartWhenAvailable reads that as a missed run and fires immediately.
    now = datetime(2026, 8, 24, 18, 0)
    assert schedule.next_occurrence(17, 0, now=now) == datetime(2026, 8, 25, 17, 0)


def test_the_trigger_never_starts_in_the_past():
    xml = _parse(schedule.build_xml(time_of_day="17:00"))
    boundary = datetime.strptime(
        _text(xml, ".//t:CalendarTrigger/t:StartBoundary"), "%Y-%m-%dT%H:%M:%S")
    assert boundary > datetime.now()


# ---------------------------------------------------------------------------
# The definition
# ---------------------------------------------------------------------------

def test_the_task_runs_step_8_daily():
    xml = _parse(schedule.build_xml(time_of_day="17:00"))

    assert _text(xml, ".//t:ScheduleByDay/t:DaysInterval") == "1"
    assert _text(xml, ".//t:CalendarTrigger/t:StartBoundary").endswith("T17:00:00")
    assert _text(xml, ".//t:Exec/t:Arguments").endswith("run.py\" --scheduled")
    assert _text(xml, ".//t:Exec/t:WorkingDirectory") == str(config.BASE_DIR)


def test_the_command_is_an_absolute_interpreter_not_the_word_python():
    # The task runs in no shell, so PATH is not a thing it has.
    command = _text(_parse(schedule.build_xml()), ".//t:Exec/t:Command")
    assert Path(command).is_absolute()
    assert Path(command).name.startswith("python")


def test_a_missed_evening_is_caught_up_by_default():
    xml = _parse(schedule.build_xml())
    assert _text(xml, ".//t:Settings/t:StartWhenAvailable") == "true"


def test_catch_up_can_be_turned_off():
    xml = _parse(schedule.build_xml(catch_up=False))
    assert _text(xml, ".//t:Settings/t:StartWhenAvailable") == "false"


def test_a_catch_up_run_cannot_land_on_top_of_a_live_one():
    xml = _parse(schedule.build_xml())
    assert _text(xml, ".//t:Settings/t:MultipleInstancesPolicy") == "IgnoreNew"


def test_the_machine_is_not_woken_by_default():
    assert _text(_parse(schedule.build_xml()), ".//t:Settings/t:WakeToRun") == "false"
    assert _text(_parse(schedule.build_xml(wake=True)),
                 ".//t:Settings/t:WakeToRun") == "true"


def test_a_hung_run_is_killed_rather_than_holding_the_day():
    xml = _parse(schedule.build_xml(time_limit_minutes=90))
    assert _text(xml, ".//t:Settings/t:ExecutionTimeLimit") == "PT90M"


def test_a_failed_run_is_retried():
    xml = _parse(schedule.build_xml(retry_minutes=30, retry_count=2))
    assert _text(xml, ".//t:RestartOnFailure/t:Count") == "2"
    assert _text(xml, ".//t:RestartOnFailure/t:Interval") == "PT30M"


def test_no_retries_omits_the_element_rather_than_asking_for_zero():
    # <Count>0</Count> is invalid, not "never".
    xml = _parse(schedule.build_xml(retry_count=0))
    assert xml.find(".//t:RestartOnFailure", NS) is None


def test_the_task_is_declared_against_the_current_user(monkeypatch):
    monkeypatch.setenv("USERDOMAIN", "DESKTOP")
    monkeypatch.setenv("USERNAME", "kuldi")
    xml = _parse(schedule.build_xml())

    assert _text(xml, ".//t:Principal/t:UserId") == "DESKTOP\\kuldi"
    assert _text(xml, ".//t:Principal/t:LogonType") == "InteractiveToken"
    assert _text(xml, ".//t:Principal/t:RunLevel") == "LeastPrivilege"


def test_a_username_with_an_ampersand_does_not_break_the_xml(monkeypatch):
    monkeypatch.setenv("USERDOMAIN", "A&B")
    monkeypatch.setenv("USERNAME", "user")
    xml = _parse(schedule.build_xml())        # parses at all = escaped properly
    assert _text(xml, ".//t:Principal/t:UserId") == "A&B\\user"


def test_the_windowless_interpreter_is_preferred(monkeypatch, tmp_path):
    real, quiet = tmp_path / "python.exe", tmp_path / "pythonw.exe"
    real.write_bytes(b"")
    quiet.write_bytes(b"")
    monkeypatch.setattr(schedule.sys, "executable", str(real))

    assert schedule.python_command() == quiet
    assert schedule.python_command(windowless=False) == real


def test_a_console_interpreter_is_used_when_there_is_no_windowless_one(
        monkeypatch, tmp_path):
    real = tmp_path / "python.exe"
    real.write_bytes(b"")
    monkeypatch.setattr(schedule.sys, "executable", str(real))
    assert schedule.python_command() == real


# ---------------------------------------------------------------------------
# Talking to schtasks
# ---------------------------------------------------------------------------

def test_install_hands_schtasks_a_utf16_definition(schtasks):
    schedule.install(task_name="TestTask")

    call = schtasks.calls[0]
    assert call["args"][:4] == ["schtasks", "/Create", "/TN", "TestTask"]
    assert "/F" in call["args"]              # installing again is how it edits
    assert call["xml"].startswith('<?xml version="1.0" encoding="UTF-16"?>')


def test_install_reports_what_schtasks_said_when_it_refuses(schtasks):
    schtasks.replies.append(FakeCompleted(1, b"", b"ERROR: Access is denied."))

    with pytest.raises(schedule.ScheduleError) as excinfo:
        schedule.install(task_name="TestTask")
    assert "Access is denied" in str(excinfo.value)


def test_install_leaves_no_temp_file_behind(schtasks):
    schedule.install(task_name="TestTask")
    temp = schtasks.calls[0]["args"][schtasks.calls[0]["args"].index("/XML") + 1]
    assert not Path(temp).exists()


def test_uninstall_treats_a_missing_task_as_the_desired_state(schtasks):
    schtasks.replies.append(
        FakeCompleted(1, b"", b"ERROR: The system cannot find the file specified.")
    )
    schedule.uninstall("TestTask")           # does not raise


def test_uninstall_still_reports_a_real_failure(schtasks):
    schtasks.replies.append(FakeCompleted(1, b"", b"ERROR: Access is denied."))
    with pytest.raises(schedule.ScheduleError):
        schedule.uninstall("TestTask")


def test_query_parses_the_fields_schtasks_lists(schtasks):
    schtasks.replies.append(FakeCompleted(0, (
        b"TaskName:      \\ReelCreatorAgent\r\n"
        b"Next Run Time: 24-08-2026 17:00:00\r\n"
        b"Status:        Ready\r\n"
        b"Last Result:   0\r\n"
    )))
    fields = schedule.query("ReelCreatorAgent")

    assert fields["next run time"] == "24-08-2026 17:00:00"
    assert fields["last result"] == "0"


def test_query_returns_nothing_when_the_task_is_not_installed(schtasks):
    schtasks.replies.append(FakeCompleted(1, b"", b"ERROR: cannot find the file"))
    assert schedule.query("Nope") is None


def test_schtasks_output_survives_a_codepage_it_cannot_decode(schtasks):
    schtasks.replies.append(FakeCompleted(1, b"ERROR: caf\xe9", b""))
    with pytest.raises(schedule.ScheduleError) as excinfo:
        schedule.install(task_name="TestTask")
    assert "ERROR" in str(excinfo.value)


def test_a_machine_without_schtasks_says_what_to_do_instead(monkeypatch):
    monkeypatch.setattr(schedule, "WINDOWS", True)

    def missing(args, **kwargs):
        raise FileNotFoundError("schtasks")

    monkeypatch.setattr(schedule.subprocess, "run", missing)
    with pytest.raises(schedule.ScheduleError) as excinfo:
        schedule.uninstall("TestTask")
    assert "preview" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Somewhere that is not Windows
# ---------------------------------------------------------------------------

def test_schtasks_is_refused_off_windows(monkeypatch):
    monkeypatch.setattr(schedule, "WINDOWS", False)
    with pytest.raises(schedule.ScheduleError) as excinfo:
        schedule.install()
    assert "crontab" in str(excinfo.value)


def test_the_crontab_line_runs_the_same_thing_at_the_same_time():
    line = schedule.crontab_line(time_of_day="17:00", python="/usr/bin/python3")
    assert line.startswith("0 17 * * *")
    assert "pipeline/run.py --scheduled" in line
    assert str(config.BASE_DIR) in line


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def test_cli_preview_prints_the_definition_and_installs_nothing(schtasks, capsys):
    assert schedule.main(["preview"]) == 0
    assert "<ScheduleByDay>" in capsys.readouterr().out
    assert schtasks.calls == []


def test_cli_preview_off_windows_prints_a_crontab_line(monkeypatch, capsys):
    monkeypatch.setattr(schedule, "WINDOWS", False)
    assert schedule.main(["preview"]) == 0
    assert "* * *" in capsys.readouterr().out


def test_cli_install_uses_the_time_it_was_given(schtasks, capsys):
    assert schedule.main(["install", "--time", "18:30"]) == 0

    assert "T18:30:00" in schtasks.calls[0]["xml"]
    assert "every day at 18:30" in capsys.readouterr().out


def test_cli_install_rejects_a_bad_time(schtasks, capsys):
    assert schedule.main(["install", "--time", "half past five"]) == 1
    assert "HH:MM" in capsys.readouterr().err
    assert schtasks.calls == []


def test_cli_console_flag_asks_for_the_visible_interpreter(schtasks, monkeypatch,
                                                           tmp_path, capsys):
    real, quiet = tmp_path / "python.exe", tmp_path / "pythonw.exe"
    real.write_bytes(b"")
    quiet.write_bytes(b"")
    monkeypatch.setattr(schedule.sys, "executable", str(real))

    schedule.main(["install", "--console"])

    assert "pythonw.exe" not in schtasks.calls[0]["xml"]
    # and the summary reports the interpreter that was actually installed
    assert "pythonw.exe" not in capsys.readouterr().out


def test_cli_install_reports_when_it_will_first_run(schtasks, capsys):
    schedule.main(["install", "--time", "17:00"])
    assert "FIRST     :" in capsys.readouterr().out


def test_cli_status_says_when_nothing_is_installed(schtasks, capsys):
    schtasks.replies.append(FakeCompleted(1, b"", b"ERROR: cannot find"))
    assert schedule.main(["status"]) == 0
    assert "is not installed" in capsys.readouterr().out


def test_cli_status_translates_the_last_result_code(schtasks, capsys):
    schtasks.replies.append(FakeCompleted(0, (
        b"Status:        Ready\r\nNext Run Time: 24-08-2026 17:00:00\r\n"
        b"Last Result:   267011\r\n"
    )))
    schedule.main(["status"])
    assert "not run yet" in capsys.readouterr().out


def test_cli_uninstall_removes_the_task(schtasks, capsys):
    assert schedule.main(["uninstall"]) == 0
    assert "/Delete" in schtasks.calls[0]["args"]
    out = capsys.readouterr().out
    assert f"removed {config.SCHEDULE_TASK_NAME}" in out
    # It used to say "Nothing will run by itself any more", which stopped being
    # true the moment there were two tasks: removing the reel task says nothing
    # about whether the read task is still installed. Claiming otherwise about
    # a scheduler is the kind of wrong that is found out weeks later.
    assert "Nothing will run by itself" not in out
    assert config.SCHEDULE_READ_TASK_NAME in out


def test_cli_run_now_starts_the_task(schtasks, capsys):
    assert schedule.main(["run-now"]) == 0
    assert "/Run" in schtasks.calls[0]["args"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))


# ---------------------------------------------------------------------------
# The reading task
# ---------------------------------------------------------------------------
#
# A second job on the same machinery: `--read` installs a task that asks the
# performance provider how already-posted reels did and writes the ledger. It
# exists because a reading has to land between PERFORMANCE_MIN_AGE_HOURS and
# PERFORMANCE_WINDOW_HOURS or that reel can never be compared with another, and
# a hand-run read does not reliably hit a 24-hour slot. What matters in these
# tests is that it is a *different task* running a *different command*, so that
# somebody who wants the numbers automated but not the video gets exactly that.

def test_read_job_runs_perform_and_not_run_py():
    xml = schedule.build_xml(job="read")
    args = ElementTree.fromstring(xml).find(".//t:Arguments", NS).text
    assert "perform.py" in args
    assert "--fetch" in args
    # The whole point of the separation: installing the reading must never be
    # a way of accidentally installing the reel.
    assert "run.py" not in args
    assert "--scheduled" not in args


def test_read_job_has_its_own_task_name():
    read = ElementTree.fromstring(schedule.build_xml(job="read"))
    reel = ElementTree.fromstring(schedule.build_xml(job="reel"))
    read_uri = read.find(".//t:URI", NS).text
    reel_uri = reel.find(".//t:URI", NS).text
    assert read_uri != reel_uri
    assert config.SCHEDULE_READ_TASK_NAME in read_uri
    assert config.SCHEDULE_TASK_NAME in reel_uri


def test_read_job_says_it_makes_nothing():
    """The description is what a person sees in Task Scheduler's own list."""
    xml = schedule.build_xml(job="read")
    description = ElementTree.fromstring(xml).find(".//t:Description", NS).text
    assert "makes no video" in description
    assert "posts nothing" in description


def test_read_job_uses_its_own_time_and_limit():
    xml = schedule.build_xml(job="read")
    root = ElementTree.fromstring(xml)
    start = root.find(".//t:StartBoundary", NS).text
    hour, minute = schedule.parse_time(config.SCHEDULE_READ_TIME)
    assert start.endswith(f"T{hour:02d}:{minute:02d}:00")
    limit = root.find(".//t:ExecutionTimeLimit", NS).text
    assert limit == f"PT{config.SCHEDULE_READ_TIME_LIMIT_MINUTES}M"


def test_read_job_catches_up_a_missed_day():
    """The setting the whole feature depends on.

    A laptop that is shut at noon is the normal case, not the exception, and a
    read that simply does not happen that day is a reel that may fall out of
    its window for good. StartWhenAvailable is what turns a missed trigger into
    a late run instead of no run.
    """
    xml = schedule.build_xml(job="read")
    root = ElementTree.fromstring(xml)
    assert root.find(".//t:StartWhenAvailable", NS).text == "true"
    # ...but never at the cost of waking the machine: there is nothing urgent
    # about a number, and the window is a day wide.
    assert root.find(".//t:WakeToRun", NS).text == "false"


def test_installing_the_reel_task_is_unchanged_by_the_read_job():
    """The reel task that existed before must come out byte-identical."""
    now = datetime(2026, 9, 16, 9, 0)
    assert (schedule.build_xml(now=now)
            == schedule.build_xml(job="reel", now=now))


def test_unknown_job_is_refused():
    with pytest.raises(schedule.ScheduleError) as exc:
        schedule.build_xml(job="numbers")
    assert "numbers" in str(exc.value)


def test_cli_read_installs_the_read_task(schtasks, capsys):
    assert schedule.main(["install", "--read"]) == 0
    created = schtasks.calls[0]
    assert "/Create" in created["args"]
    name = created["args"][created["args"].index("/TN") + 1]
    assert name == config.SCHEDULE_READ_TASK_NAME
    assert "perform.py" in created["xml"]
    out = capsys.readouterr().out
    assert "makes nothing" in out


def test_cli_read_uninstalls_only_the_read_task(schtasks):
    assert schedule.main(["uninstall", "--read"]) == 0
    args = schtasks.calls[0]["args"]
    assert args[args.index("/TN") + 1] == config.SCHEDULE_READ_TASK_NAME


def test_crontab_line_for_the_read_job():
    line = schedule.crontab_line(job="read")
    assert "perform.py --fetch" in line
    assert "run.py" not in line
