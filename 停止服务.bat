@echo off
rem ============================================================
rem  EATWHAT-model —— 停止服务（stop-service.bat）
rem
rem  按 app\artifacts\server.pid 记录的 PID 精确结束服务，
rem  查不到时再用 netstat 找出占用 8848 的进程兜底。
rem  【只结束本项目的服务，不会误伤其它 Python 程序】
rem  编码约定同 启动.bat：GBK + CRLF，不要改。
rem ============================================================
title EATWHAT-model - 停止服务
cd /d "%~dp0app"

set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY ( where py >nul 2>nul && set "PY=py" )
if not defined PY goto NOPY

%PY% stop_server.py
exit /b 0

:NOPY
echo.
echo   [错误] 没有找到 python 命令，无法自动停止服务。
echo   请打开任务管理器，手动结束占用 8848 端口的 python.exe。
echo.
pause
exit /b 1