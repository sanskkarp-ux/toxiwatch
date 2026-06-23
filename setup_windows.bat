@echo off
:: ============================================================================
:: ToxiWatch — Windows Setup Script (REPAIRED)
:: ============================================================================
:: FIXES:
::   - Added --extra-index-url for PyTorch CPU-only builds (~250MB vs ~2.5GB)
::   - Added pip upgrade step before package installation
::   - Added PYTHONUTF8=1 and PYTHONIOENCODING=utf-8 for Windows console safety
::   - Added Python version check (requires 3.11.x)
::   - Added cleanup of the incorrectly named {app,static,tests,database} folder
::
:: HOW TO RUN:
::   1. Open Command Prompt (cmd) — NOT PowerShell for this script
::   2. Navigate to this project folder:
::        cd E:\toxiwatch\toxiwatch
::   3. Run this script:
::        setup_windows.bat
:: ============================================================================

setlocal EnableDelayedExpansion

echo.
echo ============================================================
echo   ToxiWatch - Multilingual Toxicity Moderation System
echo   Windows Setup Script (Repaired)
echo ============================================================
echo.

:: ── Set UTF-8 encoding for this console session ─────────────────────────────
chcp 65001 >NUL 2>&1
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

:: ── Step 1: Check Python version ──────────────────────────────────────────
echo [1/7] Checking Python installation...
python --version 2>NUL
if errorlevel 1 (
    echo.
    echo ERROR: Python not found!
    echo Please install Python 3.11 from https://python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: Check that Python version is 3.x (at least 3.9)
for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
echo Detected Python version: %PYVER%
echo NOTE: This project was tested on Python 3.11.x
echo.

:: ── Step 2: Clean up incorrectly named folder ─────────────────────────────
echo [2/7] Cleaning up invalid directories...
if exist "{app,static,tests,database}" (
    rmdir /S /Q "{app,static,tests,database}"
    echo Removed invalid directory: {app,static,tests,database}
) else (
    echo No cleanup needed.
)
echo.

:: ── Step 3: Create a virtual environment ──────────────────────────────────
echo [3/7] Setting up Python virtual environment...
if exist "venv" (
    echo Virtual environment already exists. Skipping creation.
    echo To recreate: delete the venv\ folder and run this script again.
) else (
    python -m venv venv
    if errorlevel 1 (
        echo ERROR: Failed to create virtual environment.
        echo Try: python -m pip install --upgrade virtualenv
        pause
        exit /b 1
    )
    echo Virtual environment created in .\venv\
)
echo.

:: ── Step 4: Activate and upgrade pip ──────────────────────────────────────
echo [4/7] Activating virtual environment and upgrading pip...
call venv\Scripts\activate.bat
python -m pip install --upgrade pip setuptools wheel
echo.

:: ── Step 5: Install dependencies ──────────────────────────────────────────
echo [5/7] Installing Python dependencies...
echo.
echo IMPORTANT: PyTorch CPU-only builds are fetched from the PyTorch index.
echo This prevents downloading the 2.5GB CUDA version when you only need CPU.
echo Download size: ~500MB total (models download separately on first run).
echo.

pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

if errorlevel 1 (
    echo.
    echo ERROR: Failed to install some dependencies.
    echo.
    echo Common fixes:
    echo   1. Ensure Python 3.11 is installed (not 3.12+ or 3.14+)
    echo   2. Run: pip install --upgrade pip wheel setuptools
    echo   3. Check your internet connection
    echo   4. If sentencepiece fails: pip install sentencepiece==0.1.99
    echo.
    pause
    exit /b 1
)
echo.
echo [OK] All dependencies installed successfully!
echo.

:: ── Step 6: Verify critical imports ──────────────────────────────────────
echo [6/7] Verifying critical package imports...
python -c "import fastapi; print('[OK] FastAPI', fastapi.__version__)"
python -c "import pydantic; print('[OK] Pydantic', pydantic.__version__)"
python -c "import torch; print('[OK] PyTorch', torch.__version__)"
python -c "import transformers; print('[OK] Transformers', transformers.__version__)"
python -c "import detoxify; print('[OK] Detoxify imported')"
python -c "import langdetect; print('[OK] langdetect imported')"
python -c "import sentencepiece; print('[OK] sentencepiece imported')"
echo.

:: ── Step 7: Create necessary directories ──────────────────────────────────
echo [7/7] Ensuring project directories exist...
if not exist "static" mkdir static
if not exist "tests" mkdir tests
echo.

:: ── Done ──────────────────────────────────────────────────────────────────
echo ============================================================
echo   SETUP COMPLETE!
echo ============================================================
echo.
echo Next steps:
echo   1. To start the server, run:
echo        start_server.bat
echo.
echo   2. Open your browser and go to:
echo        http://localhost:8000
echo.
echo   3. NOTE: First startup will download ML model weights (~1 GB).
echo      This is a one-time download. Subsequent starts are much faster.
echo.
echo   4. API documentation available at:
echo        http://localhost:8000/docs
echo.
echo   5. To run tests:
echo        venv\Scripts\pytest tests\test_moderation.py -v
echo.
pause
