@echo off
setlocal enabledelayedexpansion

cd /d %~dp0

echo [1/4] Checking Python...
set PY_EXE=%CD%\.venv\Scripts\python.exe

if exist "%PY_EXE%" goto :venv_ok

echo Creating venv at .venv ...
py -3 -m venv .venv 2>nul
if errorlevel 1 (
  python -m venv .venv
)

:venv_ok
if not exist "%PY_EXE%" (
  echo ERROR: Python venv not found. Please install Python 3.8+ and rerun.
  pause
  exit /b 1
)

echo [2/4] Installing requirements (first run may take a while)...
"%PY_EXE%" -m pip install -U pip >nul
"%PY_EXE%" -m pip install -r backend\requirements.txt

echo [3/4] Starting backend on 0.0.0.0:8000 ...
start "grapetest-backend" "%PY_EXE%" -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000

echo [4/4] Starting frontend on 0.0.0.0:5173 ...
start "grapetest-frontend" cmd /c "cd /d frontend && \"%PY_EXE%\" -m http.server 5173 --bind 0.0.0.0"

echo.
echo Open in browser:
echo   http://127.0.0.1:5173/
echo Or share on LAN:
echo   http://^<your-ip^>:5173/
echo.
echo Default login: grape / 123
echo.
pause
