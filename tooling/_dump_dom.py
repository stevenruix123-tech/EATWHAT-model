# -*- coding: utf-8 -*-
"""渲染 ui_snapshot.html 并 dump DOM，找出为什么左侧表单是空的。"""
import json
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

SNAP = Path(r"C:\Users\123\Desktop\桌面应用\文档\meal-decision\artifacts\ui_snapshot.html")
DUMP = Path(r"C:\Users\123\Desktop\桌面应用\文档\_domdump.html")


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
    print("form 内 .field 个数 :", count(r'class="field"'))
    print("labels              :", count(r"<label>"))
    print("presets 按钮        :", count(r"<button[^>]*>(?:深夜加班|工作日午餐)") or "见下")
    print("chips 按钮          :", count(r'data-name="'))
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
