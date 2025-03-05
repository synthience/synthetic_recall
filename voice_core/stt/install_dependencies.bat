@echo off
echo ===================================================
echo Installing STT Dependencies
echo ===================================================
echo.

REM Change to the script directory
cd /d "%~dp0"

REM Activate virtual environment if it exists
if exist "..\..\venv\Scripts\activate.bat" (
    echo Activating virtual environment...
    call "..\..\venv\Scripts\activate.bat"
) else (
    echo No virtual environment found. Installing in global Python.
)

echo.
echo Installing packages from requirements.txt...
echo.

REM Install packages from requirements.txt
python -m pip install -r requirements.txt

echo.
echo ===================================================
echo Installation complete!
echo ===================================================
echo.
echo To verify the installation, run: python test_imports.py
echo.

pause
