@echo off
title Retinal Imaging and Ophthalmic Screening System
echo ====================================================================
echo      Retinal Imaging and Ophthalmic Screening System
echo ====================================================================
echo.

:: Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo [ERROR] Virtual environment (.venv) not found.
    echo Please make sure you have run your environment setup.
    echo attempting to run using global python...
    echo.
    goto run_global
)

echo [INFO] Activating virtual environment...
call .venv\Scripts\activate.bat
goto run_app

:run_global
:: Check if python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in the PATH.
    echo Please install Python 3.10+ and add it to your PATH variables.
    pause
    exit /b 1
)

:run_app
echo [INFO] Opening the browser...
:: Wait a tiny bit then open the browser so the server has a moment to initialize
start "" http://127.0.0.1:8000

echo [INFO] Starting FastAPI / Uvicorn Server...
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

if errorlevel 1 (
    echo.
    echo [ERROR] Server failed to start or was stopped unexpectedly.
    echo Please ensure dependencies in requirements.txt are installed.
)
pause
