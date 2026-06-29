@echo off
setlocal
cd /d "%~dp0"
set /p AUTH="Enter auth as user:password (recommended): "
if not exist ".agent_runtime\venv\Scripts\python.exe" (
  py -3 install.py --torch auto
)
".agent_runtime\venv\Scripts\python.exe" run_app.py --lan --port 7860 --auth %AUTH%
pause
