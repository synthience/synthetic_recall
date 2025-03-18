@echo off
echo Testing Emotional Memory Integration
echo ==================================

REM Check if Python is in PATH
where python >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo Error: Python not found in PATH
    exit /b 1
)

REM Set the Python path to the project root
set PYTHONPATH=%PYTHONPATH%;%~dp0..

REM Check for --tool-test flag
if "%1"=="--tool-test" (
    echo Running tool integration test...
    python %~dp0..\utils\test_emotional_memory.py --tool-test
) else (
    echo Running memory integration test...
    python %~dp0..\utils\test_emotional_memory.py
)

echo.
echo Test completed
