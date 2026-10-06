# -*- coding: utf-8 -*-
"""
preflight.py —— 启动前环境自检（供 启动.bat 调用）

为什么需要它：启动脚本原本只检查 `python` 命令在不在，不检查 numpy / pandas。
在一台只装了 Python、没装依赖的电脑上，双击启动会以一堆 ImportError 堆栈结束，
用户根本看不出该做什么。这里把「缺什么、怎么补」用中文直说。

退出码：0 = 环境可用；1 = 环境不可用（启动脚本据此停下）
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEED_PY = (3, 10)


def main() -> int:
    print("=" * 62)
    print("  吃什么 V1beta —— 运行环境自检")
    print("=" * 62)

    ver = sys.version_info
    print(f"  Python 版本 : {sys.version.split()[0]}   ({sys.executable})")

    if ver < NEED_PY:
        print()
        print(f"  [错误] 需要 Python {NEED_PY[0]}.{NEED_PY[1]} 或更高版本。")
        print("         请到 https://www.python.org/downloads/ 安装新版 Python，")
        print("         安装时务必勾选 “Add python.exe to PATH”。")
        return 1

    missing: list[str] = []
    for mod, pkg in (("numpy", "numpy"), ("pandas", "pandas")):
        try:
            m = __import__(mod)
            v = getattr(m, "__version__", "?")
            print(f"  {mod:<11} : 已安装 {v}")
        except Exception as e:                       # noqa: BLE001
            print(f"  {mod:<11} : [缺失] {e}")
            missing.append(pkg)

    # 顺带确认模型与数据在位（这两个缺了同样起不来，提前说清楚）
    need_files = [
        ("模型", HERE / "artifacts" / "model.npz"),
        ("菜品库", HERE / "artifacts" / "dishes.csv"),
        ("样本数据", HERE / "artifacts" / "samples.csv"),
    ]
    for label, p in need_files:
        print(f"  {label:<9} : {'OK' if p.exists() else '[缺失] ' + str(p)}")
        if not p.exists():
            missing.append(p.name)

    if missing:
        print()
        print("  [错误] 环境不完整，缺少：" + "、".join(dict.fromkeys(missing)))
        print()
        print("  解决办法（任选其一）：")
        print("    1) 联网时双击本目录下的  安装依赖.bat  （自动执行 pip install）")
        print("    2) 手动执行：")
        print("         python -m pip install -r requirements.txt")
        print("    3) 若 pip 太慢，可加国内镜像：")
        print("         python -m pip install -r requirements.txt "
              "-i https://pypi.tuna.tsinghua.edu.cn/simple")
        print()
        return 1

    print()
    print("  环境自检通过，开始启动服务 ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
