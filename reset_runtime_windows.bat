@echo off
cd /d "%~dp0"
echo This will delete packages, models, and caches under .agent_runtime.
pause
rmdir /s /q .agent_runtime
pause
