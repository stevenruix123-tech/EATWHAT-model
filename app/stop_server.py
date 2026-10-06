# -*- coding: utf-8 -*-
"""
stop_server.py —— 停止后台服务

两种方式一起用，保证一定能停掉：
  1. 读 artifacts/server.pid（启动器记录的 PID），确认它确实是我们的服务后终止
  2. 兜底：查出谁在监听 8848 端口，终止它
只杀「监听本项目端口」或「PID 文件里记录的」进程，不会误伤其它 python 程序。
"""

from __future__ import annotations

import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PORT = 8848
HOST = "127.0.0.1"
PID_FILE = HERE / "artifacts" / "server.pid"


def port_open() -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.4)
        return s.connect_ex((HOST, PORT)) == 0


def pid_listening_on_port() -> int | None:
    """用 netstat 找出监听该端口的进程 PID（不依赖 psutil）"""
    try:
        out = subprocess.run(["netstat", "-ano", "-p", "TCP"],
                             capture_output=True, text=True,
                             encoding="utf-8", errors="replace").stdout
    except Exception:
        return None
    for line in out.splitlines():
        m = re.search(rf"TCP\s+\S*:{PORT}\s+\S+\s+LISTENING\s+(\d+)", line)
        if m:
            return int(m.group(1))
    return None


def kill(pid: int) -> bool:
    r = subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"],
                       capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    ok = r.returncode == 0
    if not ok:
        # taskkill 失败时退回 os.kill（在 Windows 上等价于 TerminateProcess）
        try:
            os.kill(pid, 9)
            ok = True
        except Exception:
            ok = False
    return ok


def main() -> int:
    print("=" * 56)
    print("  今天该吃什么 —— 停止服务")
    print("=" * 56)

    targets: list[int] = []
    if PID_FILE.exists():
        try:
            pid = int(PID_FILE.read_text(encoding="utf-8").strip())
            if pid > 0 and pid != os.getpid():
                targets.append(pid)
        except Exception:
            pass

    net_pid = pid_listening_on_port()
    if net_pid and net_pid not in targets and net_pid != os.getpid():
        targets.append(net_pid)

    if not targets:
        if port_open():
            print("  端口被占用但查不到 PID，请手动结束。")
            input("\n  按回车关闭...")
            return 1
        print("  服务本来就没有在运行。")
        PID_FILE.unlink(missing_ok=True)
        time.sleep(1.2)
        return 0

    for pid in targets:
        print(f"  正在停止 PID {pid} ...")
        ok = kill(pid)
        print(f"    {'成功' if ok else '失败（可能已退出）'}")

    # 等操作系统释放端口
    for _ in range(20):
        if not port_open():
            break
        time.sleep(0.25)

    PID_FILE.unlink(missing_ok=True)
    if port_open():
        print("\n  端口仍在监听，可能还有其它进程占用 8848。")
        print("  可改用别的端口启动： python server.py --port 9000")
        input("\n  按回车关闭...")
        return 1
    print("\n  服务已停止，端口已释放。")
    time.sleep(1.2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
