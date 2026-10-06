@echo off
rem ============================================================
rem  EATWHAT-model —— 双击启动（start.bat）
rem
rem  作用：起本地服务 + 自动打开浏览器。
rem  关掉这个窗口【不会】停止服务；要停止请双击 停止服务.bat。
rem
rem  关于编码：本文件必须是【GBK 编码 + CRLF 换行】，两者都不能改。
rem  原因：cmd.exe 按系统 OEM 代码页（简中 Windows = 936/GBK）解析 .bat。
rem  若存成 UTF-8，中文字节会被错拆，cmd 会把半个汉字当成命令，
rem  报出 "'.exe' is not recognized" 这类莫名其妙的错 —— 本项目踩过。
rem  LF-only 换行同样不行：cmd 不认识 LF，会把多行粘成一行。
rem ============================================================
title EATWHAT-model - 启动中
cd /d "%~dp0app"

set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY ( where py >nul 2>nul && set "PY=py" )
if not defined PY goto NOPY

rem ---- 启动前自检：Python 版本 + numpy/pandas + 模型与数据是否齐全 ----
%PY% preflight.py
if errorlevel 1 goto FAIL

rem ---- 启动本身（launch.py 负责：防重复启动、等服务就绪、开浏览器、记 PID）----
%PY% launch.py
if errorlevel 1 goto FAIL

rem 停 5 秒让人看清提示，然后自动关窗（服务仍在后台运行）
rem 用 ping 而不是 timeout：timeout 在输入被重定向时会直接报错
ping -n 5 127.0.0.1 >nul
exit /b 0

:NOPY
echo.
echo   [错误] 没有找到 python 命令。
echo.
echo   请先安装 Python 3.10 或更高版本：
echo       https://www.python.org/downloads/
echo   安装时务必勾选 "Add python.exe to PATH"，装完重开本窗口再双击一次。
echo.
pause
exit /b 1

:FAIL
echo.
echo   启动失败。常见原因：
echo     1. 没装 numpy / pandas  --^> 双击 安装依赖.bat
echo     2. 端口 8848 已被占用   --^> 先双击 停止服务.bat
echo     3. 其它原因             --^> 看上面输出，或 app\artifacts\server.log
echo.
pause
exit /b 1