@echo off
rem Opens the GPU window for 25 seconds with music, saves a screenshot, then closes.
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo Opening the claude fm window for 25 seconds... (details in window.log)
> window.log echo === %date% %time%
>> window.log uv run python -m claudefm --seconds 25 --screenshot window.png 2>&1
>> window.log echo === exit code %errorlevel% %date% %time%
