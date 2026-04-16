@echo off
REM start_all.bat — Launch all model API servers dynamically.
REM Reads config from start_all.cfg if present, otherwise auto-detects conda.
REM Scans checkpoints\ for any *_best_model.pth files and launches each one.

setlocal EnableDelayedExpansion
set BACKEND=%~dp0
set CFG=%BACKEND%start_all.cfg

REM ── 1. Load or create config ─────────────────────────────────────────────────
if exist "%CFG%" (
    for /f "usebackq tokens=1,* delims==" %%A in ("%CFG%") do (
        set "%%A=%%B"
    )
    echo [config] Loaded settings from start_all.cfg
) else (
    echo [config] No start_all.cfg found — auto-detecting...

    REM Auto-detect conda root (checks common locations)
    for %%P in (
        "%USERPROFILE%\miniconda3"
        "%USERPROFILE%\anaconda3"
        "%LOCALAPPDATA%\miniconda3"
        "%LOCALAPPDATA%\anaconda3"
        "C:\miniconda3"
        "C:\anaconda3"
        "C:\ProgramData\miniconda3"
        "C:\ProgramData\anaconda3"
    ) do (
        if not defined CONDA_ROOT (
            if exist "%%~P\Scripts\activate.bat" set "CONDA_ROOT=%%~P"
        )
    )

    if not defined CONDA_ROOT (
        echo [error] Could not find a conda installation.
        echo         Set CONDA_ROOT manually in start_all.cfg
        pause & exit /b 1
    )

    REM Default env name — override in start_all.cfg
    set "ENV_NAME=cxr"

    REM Save discovered values for next time
    (
        echo CONDA_ROOT=!CONDA_ROOT!
        echo ENV_NAME=cxr
    ) > "%CFG%"
    echo [config] Created start_all.cfg — edit it to change settings.
)

set "ACTIVATE=%CONDA_ROOT%\Scripts\activate.bat"

if not exist "%ACTIVATE%" (
    echo [error] activate.bat not found at: %ACTIVATE%
    echo         Check CONDA_ROOT in start_all.cfg
    pause & exit /b 1
)

REM ── 2. Port assignment — first model found gets 5001, next 5002, etc. ────────
set /a PORT=5001
set LAUNCHED=0

REM ── 3. Scan checkpoints\ for *_best_model.pth and launch each ────────────────
if not exist "%BACKEND%checkpoints\" (
    echo [error] No checkpoints\ folder found at: %BACKEND%checkpoints\
    pause & exit /b 1
)

for %%F in ("%BACKEND%checkpoints\*_best_model.pth") do (
    REM Extract model name: strip path and _best_model.pth suffix
    set "FNAME=%%~nF"
    set "MODEL=!FNAME:_best_model=!"

    echo [launch] !MODEL! on port !PORT! ^(%%~nxF^)
    start "!MODEL! :!PORT!" cmd /k "call "%ACTIVATE%" %ENV_NAME% && cd /d "%BACKEND%" && python api.py --weights checkpoints\%%~nxF --model !MODEL! --port !PORT!"

    set /a PORT+=1
    set /a LAUNCHED+=1
)

echo.
if %LAUNCHED%==0 (
    echo [warn] No *_best_model.pth files found in checkpoints\
    echo        Nothing was launched.
) else (
    echo %LAUNCHED% server(s) launched. Minimise the windows and leave them running.
)

endlocal
pause
