@echo off
echo ====================================
echo CV Validator Application Starting...
echo ====================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python එක install කරලා නෑ!
    echo කරුණාකර Python install කරන්න: https://www.python.org/downloads/
    pause
    exit /b
)

REM Check if requirements are installed
echo Checking dependencies...
pip show Flask >nul 2>&1
if errorlevel 1 (
    echo Installing requirements...
    pip install -r requirements.txt
)

echo.
echo Starting CV Validator...
echo Open your browser and go to: http://127.0.0.1:5000
echo.
echo Press Ctrl+C to stop the server
echo.

python cv_validator_app.py

pause
