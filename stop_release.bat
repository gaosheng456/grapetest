@echo off
setlocal enabledelayedexpansion

cd /d %~dp0

echo Stopping processes listening on ports 8000 and 5173...

for %%P in (8000 5173) do (
  for /f "tokens=5" %%a in ('netstat -ano ^| findstr ":%%P" ^| findstr LISTENING') do (
    echo Killing PID %%a (port %%P)
    taskkill /F /PID %%a >nul 2>nul
  )
)

echo Done.
pause
