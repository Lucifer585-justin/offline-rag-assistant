@echo off
chcp 65001 >nul
title Offline RAG Assistant
setlocal

set ROOT=%~dp0
set PYTHONIOENCODING=utf-8

echo ============================================================
echo   Offline RAG Assistant
echo   本地大模型 + 本地知识库 + 多轮记忆，全程不联网
echo ============================================================
echo.

echo [1/4] 检查 Ollama...
powershell -NoProfile -Command "try { $v=(Invoke-WebRequest 'http://127.0.0.1:11434/api/version' -UseBasicParsing -TimeoutSec 3).Content; Write-Host ('      OK ' + $v) -ForegroundColor Green } catch { Write-Host '      未运行，尝试启动...' -ForegroundColor Yellow; Start-Process \"$env:LOCALAPPDATA\Programs\Ollama\ollama.exe\" -ArgumentList 'serve' -WindowStyle Hidden; Start-Sleep 8 }"

echo [2/4] 检查模型...
"%LOCALAPPDATA%\Programs\Ollama\ollama.exe" list

echo.
echo [3/4] 推理设备检查（应为 100%% GPU）...
"%LOCALAPPDATA%\Programs\Ollama\ollama.exe" ps

echo.
echo [4/4] 启动服务，手机/其他设备可访问：
powershell -NoProfile -Command "Get-NetIPAddress -AddressFamily IPv4 -ErrorAction SilentlyContinue | Where-Object { $_.IPAddress -notlike '127.*' -and $_.IPAddress -notlike '169.254.*' -and $_.InterfaceAlias -notmatch 'VMware|Loopback' } | ForEach-Object { Write-Host ('      http://' + $_.IPAddress + ':5000') -ForegroundColor Cyan }"
echo.

if exist "%ROOT%.venv\Scripts\python.exe" (
    set PY=%ROOT%.venv\Scripts\python.exe
) else (
    set PY=python
)

"%PY%" "%ROOT%app.py" %*
pause
