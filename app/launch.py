# -*- coding: utf-8 -*-
"""
launch.py —— 双击启动器：起服务 + 自动开浏览器

行为：
  1. 先探测端口。已经在跑 → 不重复启动，直接打开浏览器
  2. 没在跑 → 后台启动 server.py --no-browser，等它就绪，再打开浏览器
  3. 把子进程 PID 记到 artifacts/server.pid，方便 stop_server.py 精确停止

为什么不让 server.py 自己开浏览器：那样「先启动、后打开」的时序不可控，
而且重复双击会起两个服务。这里统一由启动器负责。
"""

from __future__ import annotations

import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = 8848
HOST = "127.0.0.1"
URL = f"http://{HOST}:{PORT}/"
PID_FILE = HERE / "artifacts" / "server.pid"
LOG_FILE = HERE / "artifacts" / "server.log"


def port_open(host: str = HOST, port: int = PORT, timeout: float = 0.4) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        return s.connect_ex((host, port)) == 0


def wait_ready(seconds: float = 40.0) -> bool:
    """等 HTTP 真正能响应（不是只看端口在监听）"""
    deadline = time.time() + seconds
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(URL, timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.5)
    return False


def main() -> int:
    print("=" * 56)
    print("  今天该吃什么 —— 启动中")
    print("=" * 56)

    if port_open():
        print(f"  服务已经在运行，直接打开界面。")
        webbrowser.open(URL)
        print(f"  地址： {URL}")
        print("\n  这个窗口可以关掉，服务不会停。")
        return 0

    if not (HERE / "artifacts" / "model.npz").exists():
        print("  找不到 artifacts/model.npz")
        print("  请先运行： python build_dataset.py  然后  python train.py")
        input("\n  按回车关闭...")
        return 1

    print("  正在启动后台服务 ...")
    log = open(LOG_FILE, "a", encoding="utf-8")
    log.write(f"\n\n===== 启动 {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    log.flush()
    # 用 pythonw 之外的方式：保留 python，但把输出重定向到日志文件，
    # 这样窗口关掉也不会因为写 stdout 失败而崩。
    proc = subprocess.Popen(
        [sys.executable, str(HERE / "server.py"),
         "--port", str(PORT), "--no-browser"],
        cwd=str(HERE), stdout=log, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
    )
    PID_FILE.parent.mkdir(exist_ok=True)
    PID_FILE.write_text(str(proc.pid), encoding="utf-8")

    if not wait_ready():
        print("  启动超时或失败，最近日志：")
        try:
            tail = LOG_FILE.read_text(encoding="utf-8", errors="replace").splitlines()[-12:]
            for line in tail:
                print("    " + line)
        except Exception:
            pass
        input("\n  按回车关闭...")
        return 1

    print(f"  服务已就绪（PID {proc.pid}）")
    webbrowser.open(URL)
    print(f"\n  界面地址： {URL}")
    print("  停止服务： 双击「停止服务.bat」")
    print("\n  这个窗口可以关掉，服务会在后台继续运行。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
