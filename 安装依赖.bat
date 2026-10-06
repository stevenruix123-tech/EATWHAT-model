@echo off
rem ============================================================
rem  EATWHAT-model —— 安装依赖（install-deps.bat）
rem
rem  首次在新电脑上使用、或提示缺少 numpy / pandas 时，双击本文件。
rem  需要联网。编码约定同 启动.bat：GBK + CRLF，不要改。
rem ============================================================
title EATWHAT-model - 安装依赖
cd /d "%~dp0"

set "PY="
where python >nul 2>nul && set "PY=python"
if not defined PY ( where py >nul 2>nul && set "PY=py" )
if not defined PY goto NOPY

echo.
echo   正在安装 numpy + pandas（需要联网，约 20-60 MB）...
echo.
%PY% -m pip install -r requirements.txt
if errorlevel 1 goto MIRROR

echo.
echo   安装完成。现在可以双击 启动.bat 了。
echo.
pause
exit /b 0

:MIRROR
echo.
echo   直连安装失败，改用国内镜像重试（清华 TUNA）...
echo.
%PY% -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo.
echo   若仍然失败，请手动执行上面第二条命令，或检查网络/代理设置。
echo.
pause
exit /b 0

:NOPY
echo.
echo   [错误] 没有找到 python 命令。
echo.
echo   请先安装 Python 3.10 或更高版本：
echo       https://www.python.org/downloads/
echo   安装时务必勾选 "Add python.exe to PATH"。
echo.
pause
exit /b 1