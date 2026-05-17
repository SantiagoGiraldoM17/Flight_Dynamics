@echo off
REM ============================================================
REM   RCAM Flight Dynamics  -  Dependency installer (Windows)
REM   Creates a local virtual environment in .\venv and installs
REM   everything from requirements.txt into it.
REM ============================================================

setlocal
cd /d "%~dp0"

echo.
echo [1/3] Checking Python...
where python >nul 2>nul
if errorlevel 1 (
    echo ERROR: Python is not on PATH. Install Python 3.8+ from python.org
    echo         and tick "Add Python to PATH" during setup.
    pause
    exit /b 1
)
python --version

echo.
echo [2/3] Creating virtual environment in .\venv ...
if not exist "venv" (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: failed to create virtual environment.
        pause
        exit /b 1
    )
) else (
    echo     venv already exists - skipping.
)

echo.
echo [3/3] Installing dependencies from requirements.txt ...
call "venv\Scripts\activate.bat"
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo ERROR: pip install failed.
    pause
    exit /b 1
)

echo.
echo ============================================================
echo   Install complete.  Double-click run.bat to start the sim.
echo ============================================================
pause
endlocal
