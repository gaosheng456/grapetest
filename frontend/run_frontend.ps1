$ErrorActionPreference = 'Stop'

Set-Location $PSScriptRoot

# 简单静态服务器（避免 file:// 带来的 CORS/安全限制）
$python = Join-Path $PSScriptRoot '..\.venv\Scripts\python.exe'
& $python -m http.server 5173
