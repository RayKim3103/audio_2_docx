@echo off
setlocal
cd /d "%~dp0"
if not exist ".agent_runtime\venv\Scripts\python.exe" (
  echo [setup] Creating local isolated runtime...
  py -3 install.py --torch auto
  if errorlevel 1 (
    echo [setup] py launcher failed. Trying python...
    python install.py --torch auto
  )
)
".agent_runtime\venv\Scripts\python.exe" run_app.py --host 127.0.0.1 --port 7860
pause
