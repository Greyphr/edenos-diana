@echo off
setlocal

rem enroll_voice.bat - one-click setup for Eden's voice enrollment.
rem Double-click this file (or run it from a command prompt) to teach Eden
rem your voice. No Python or terminal knowledge needed.

cd /d "%~dp0"

echo.
echo ============================================================
echo  EDEN - VOICE ENROLLMENT
echo ============================================================
echo.
echo  This will learn your voice so Eden can recognize you.
echo  You'll be asked to say a few phrases clearly into the mic.
echo.
echo ============================================================
echo.

if not exist ".venv\Scripts\activate.bat" (
    echo  The virtual environment (.venv) is missing.
    echo  Run the setup steps below once, then double-click this file again:
    echo.
    echo      python -m venv .venv
    echo      .venv\Scripts\python -m pip install -r requirements.txt
    echo.
    echo  (You only need to do this once on a new machine.)
    echo.
    pause
    exit /b 1
)

call ".venv\Scripts\activate.bat"

echo  Starting enrollment...
echo.
".venv\Scripts\python.exe" -m identity.enroll

echo.
echo  Enrollment finished - you can close this window.
echo.
pause
endlocal