$ErrorActionPreference = 'Stop'

Set-Location (Join-Path $PSScriptRoot '..')

$python = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'

& $python -m uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
