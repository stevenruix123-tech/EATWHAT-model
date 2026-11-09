# -*- coding: utf-8 -*-
"""渲染 app/artifacts/ui_snapshot.html 并把渲染后的 DOM dump 到同目录 _domdump.html。

用途：核对快照到底渲染出了什么（表单空不空、预设选中没、忌口按钮多少个）。
路径一律从脚本位置推仓库根，别写死绝对路径（这项目被移动过目录）。
"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # tooling/ 的上一级 = 仓库根
SNAP = ROOT / "app" / "artifacts" / "ui_snapshot.html"
DUMP = ROOT / "app" / "artifacts" / "_domdump.html"


def find_chrome():
    for p in [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]:
        if Path(p).exists():
            return p
    for n in ("chrome", "msedge", "chromium"):
        w = shutil.which(n)
        if w:
            return w
    return None


def main():
    exe = find_chrome()
    if not exe:
        print("找不到 Chrome/Edge")
        return 1
    prof = tempfile.mkdtemp(prefix="domdump_")
    cmd = [exe, "--headless=new", "--disable-gpu", "--no-sandbox",
           f"--user-data-dir={prof}", "--virtual-time-budget=6000",
           "--dump-dom", SNAP.as_uri()]
    r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8",
                       errors="replace", timeout=120)
    dom = r.stdout or ""
    DUMP.write_text(dom, encoding="utf-8")
    print(f"DOM 字节: {len(dom)}  -> {DUMP}")

    def count(pat):
        return len(re.findall(pat, dom))

    print()
    print("form 内 .field 个数 :", count(r'class="field[ "]'))   # 含 class="field wide"
    print("labels              :", count(r"<label>"))
    m_pre = re.search(r'id="presets">(.*?)</div>', dom, re.S)
    print("presets 按钮        :",
          len(re.findall(r"<button", m_pre.group(1))) if m_pre else 0)
    print("chips 按钮          :", count(r'data-name="'))
    print("原始打分表行数      :", count(r"<tr>"))
    print()
    m = re.search(r'<div id="form">(.*?)</div>\s*<div id="chips">', dom, re.S)
    if not m:
        m = re.search(r'<div id="form">(.{0,400})', dom, re.S)
    print("--- #form 开头 500 字符 ---")
    print((m.group(1)[:500] if m else "未匹配到 #form"))
    print()
    # 标签文字检查
    for kw in ("时段", "饥饿程度", "可用时间", "忌口", "无忌口"):
        print(f"  DOM 含 {kw!r}: {kw in dom}")
    print()
    print("stderr 摘要:", (r.stderr or "")[-400:])
    return 0


if __name__ == "__main__":
    sys.exit(main())
