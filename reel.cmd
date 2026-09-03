@echo off
REM ---------------------------------------------------------------------------
REM One command for the whole agent: `reel` scouts a topic, makes the reel and
REM writes its caption - step 8, exactly as `python pipeline/run.py` does.
REM
REM Everything after `reel` is handed to run.py untouched, so:
REM     reel                          scout, choose, make it
REM     reel --dry-run                choose a topic, make nothing
REM     reel --status                 how the last runs went
REM     reel "why the ocean is salty" -d 40
REM
REM %~dp0 is this file's own folder, so the paths below do not care where the
REM shell happens to be - the project can be moved, or this put on PATH, and it
REM still runs the right run.py. config.py resolves the rest off its own
REM __file__, so there is no directory to change into first.
REM ---------------------------------------------------------------------------
setlocal
python "%~dp0pipeline\run.py" %*
REM Pass run.py's exit code out: 0 for a reel or a deliberate skip, 1 for a
REM failure. A scheduler - and `if errorlevel` in any shell - reads this.
exit /b %ERRORLEVEL%
