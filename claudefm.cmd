@echo off
rem Run claude fm in its GPU window. Keys: n scene, t track, c time of day, r look, f fullscreen, q quit.
cd /d "%~dp0"
uv run python -m claudefm %*
