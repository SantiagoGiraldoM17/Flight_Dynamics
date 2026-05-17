@echo off
REM ============================================================
REM   RCAM Flight Dynamics  -  Launcher (Windows)
REM   Activates the local venv and runs flight_main.py.
REM ============================================================

setlocal
cd /d "%~dp0"

if not exist "venv\Scripts\activate.bat" (
    echo ERROR: virtual environment not found.
    echo         Run install.bat first.
    pause
    exit /b 1
)

call "venv\Scripts\activate.bat"
python flight_main.py

REM Keep the window open if Python crashes so the error is readable
if errorlevel 1 pause
endlocal
