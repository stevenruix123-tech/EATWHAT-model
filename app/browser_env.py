# -*- coding: utf-8 -*-
"""
browser_env.py —— headless Chrome/Edge 的定位与 --dump-dom 封装

被两个脚本共用，避免 Chrome 参数在两边各写一份、各自漂移：
  - app/snapshot_ui.py：把真实页面渲染成静态快照
  - app/smoke_ui.py   ：DOM 断言（点按钮、量位置、读 aria）

这里只用标准库，不 import numpy/pandas，所以单独跑也不会拖模型。
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

CHROME_PATHS = (
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
)


def find_chrome() -> str | None:
    for p in CHROME_PATHS:
        if Path(p).exists():
            return p
    for name in ("chrome", "msedge", "chromium"):
        w = shutil.which(name)
        if w:
            return w
    return None


def dump_dom(exe: str, url: str, budget_ms: int = 20000) -> tuple[str, str]:
    """渲染 url 并返回 (DOM, stderr)。

    stdout/stderr 走管道；stdout 为空时，stderr 里能看出是「沙箱拦了命名管道」
    还是「页面本身崩了」。
    """
    prof = tempfile.mkdtemp(prefix="ds_dom_")
    try:
        cmd = [exe, "--headless=new", "--disable-gpu", "--no-sandbox",
               "--no-first-run", "--disable-extensions", "--hide-scrollbars",
               f"--user-data-dir={prof}", f"--virtual-time-budget={budget_ms}",
               "--window-size=1280,900", "--dump-dom", url]
        r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                           errors="replace", timeout=180)
        return r.stdout or "", r.stderr or ""
    finally:
        shutil.rmtree(prof, ignore_errors=True)


def first_error_line(stderr: str) -> str:
    """从 Chrome 的 stderr 里挑一行最能说明问题的。"""
    for line in stderr.splitlines():
        line = line.strip()
        if line and ("FATAL" in line or "ERROR" in line or "denied" in line.lower()
                     or "拒绝" in line):
            return re.sub(r"^\[[^\]]*\]\s*", "", line)[:90]
    lines = [ln for ln in stderr.strip().splitlines() if ln.strip()]
    return lines[0][:90] if lines else ""
