@echo off
REM Change to the script directory
cd /d "%~dp0"

REM Set PYTHONPATH to the parent directory
set "PYTHONPATH=%~dp0.."

REM Activate the virtual environment
call ".venv\Scripts\activate.bat"

echo Environment activated. PYTHONPATH set.
echo You can now run: python -m stress_detection.main --mode test_run
cmd /k
