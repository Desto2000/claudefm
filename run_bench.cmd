@echo off
rem Sets up the project environment with uv (a .venv in this folder, nothing global)
rem and times every scene and look on the GPU. Everything goes to bench.log.
cd /d "%~dp0"
set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
echo Benchmarking claude fm on the GPU... (details in bench.log)
> bench.log echo === %date% %time%
>> bench.log uv --version 2>&1
>> bench.log nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1
>> bench.log echo === uv sync
>> bench.log uv sync 2>&1
>> bench.log echo === warp
>> bench.log uv run python -c "import warp as wp; wp.init(); print('devices', wp.get_devices())" 2>&1
>> bench.log echo === bench 4K canvas, 8x16 cells
>> bench.log uv run python -m claudefm --bench 120 2>&1
>> bench.log echo === bench 2560x1440 canvas, 6x12 cells
>> bench.log uv run python -m claudefm --bench 120 --canvas 2560x1440 --cell 6x12 2>&1
>> bench.log echo === done %date% %time%
