@echo off
:: ============================================================================
:: ToxiWatch — Start Server (Windows, REPAIRED)
:: ============================================================================
:: FIXES:
::   - Added PYTHONUTF8=1 and PYTHONIOENCODING=utf-8 to prevent Unicode errors
::     in Windows console when ML models log emoji/Unicode characters
::   - Added chcp 65001 to set console to UTF-8 code page
::   - Added check to ensure venv exists before trying to activate it
::
:: Usage: Double-click OR run from Command Prompt:
::   start_server.bat
:: ============================================================================

@echo off
chcp 65001 >NUL 2>&1

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8

echo.
echo ============================================================
echo   ToxiWatch - Multilingual Toxicity Moderation System
echo   Starting Server...
echo ============================================================
echo.

:: Check venv exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup_windows.bat first.
    pause
    exit /b 1
)

echo Activating virtual environment...
call venv\Scripts\activate.bat

echo.
echo Starting ToxiWatch server on http://localhost:8000
echo API Docs: http://localhost:8000/docs
echo.
echo NOTE: First startup downloads ML model weights (~1 GB). Please wait.
echo       This is a one-time download. Subsequent starts take ~30 seconds.
echo.
echo Press Ctrl+C to stop the server.
echo.

:: --reload: auto-restart when Python files change (development mode)
:: --host 0.0.0.0: accessible from other devices on your network
:: --port 8000: default port
uvicorn main:app --reload --host 0.0.0.0 --port 8000

pause
